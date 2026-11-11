from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from long_video_pipeline.job_manager import JobContext, atomic_write_text
from long_video_pipeline.splitter import split_video, probe_duration, ChunkInfo, is_valid_audio_chunk
from long_video_pipeline.transcription_stage import run_transcription_stage
from long_video_pipeline.translation_stage import run_translation_stage
from long_video_pipeline.dubbing_stage import run_dubbing_stage
from long_video_pipeline.splitter import load_manifest
from modules.video_assembler import VideoAssembler
from modules.srt_parser import SRTParser
from config import MAX_SPEED_FACTOR
import modules.time_sync as time_sync_module

SOURCE_MAP = {"auto": None, "en": "en", "fr": "fr", "es": "es", "de": "de", "it": "it", "pt": "pt", "ja": "ja", "ko": "ko", "zh": "zh", "ru": "ru", "ar": "ar", "hi": "hi"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}


def timestamp(seconds: float) -> str:
    total = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(total, 3600000); m, rem = divmod(rem, 60000); s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def seed_external_srt(job, srt_path: Path, chunks: list[ChunkInfo]) -> int:
    """Create the pipeline checkpoints from an existing, timed SRT file."""
    segments = SRTParser().parse_srt(str(srt_path))
    if not segments:
        raise RuntimeError(f"External SRT contains no readable cues: {srt_path}")
    total = 0
    for chunk in chunks:
        chunk_segments = []
        for item in segments:
            start = max(float(item["start"]), float(chunk.start))
            end = min(float(item["end"]), float(chunk.end))
            if end <= start:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            row = {
                "index": len(chunk_segments) + 1,
                "start": start - float(chunk.start),
                "end": end - float(chunk.start),
                "text": text,
                "normal_text": text,
                "fitted_text": text,
                "translated_text": text,
            }
            chunk_segments.append(row)
        if not chunk_segments:
            continue
        payload = {"segments": chunk_segments, "language": "external-srt"}
        transcription_path = job.path_for_chunk(job.transcription_dir, chunk.number, ".json")
        translation_path = job.path_for_chunk(job.translation_dir, chunk.number, ".json")
        transcription_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        translation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        chunk_srt = job.transcription_dir / f"chunk_{chunk.number:03d}.srt"
        chunk_srt_text = "\n".join(
            f"{i}\n{timestamp(row['start'])} --> {timestamp(row['end'])}\n{row['text']}\n"
            for i, row in enumerate(chunk_segments, 1)
        )
        chunk_srt.write_text(chunk_srt_text, encoding="utf-8")
        for suffix in ("_normal.srt", "_fitted.srt"):
            (job.translation_dir / f"chunk_{chunk.number:03d}{suffix}").write_text(chunk_srt_text, encoding="utf-8")
        total += len(chunk_segments)
    if total <= 0:
        raise RuntimeError("External SRT has no cues inside the input media duration.")
    return total


def write_full_srt(job, target_path: Path, key: str = "text") -> int:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target_path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in load_manifest(job):
            transcription_path = job.transcription_dir / f"chunk_{chunk.number:03d}.json"
            translation_path = job.translation_dir / f"chunk_{chunk.number:03d}.json"
            candidates = (transcription_path, translation_path) if key == "text" else (translation_path, transcription_path)
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for seg in data.get("segments", []):
                text = str(seg.get(key, "")).strip()
                if not text and key != "text":
                    text = str(seg.get("translated_text", "")).strip()
                if text:
                    start = float(seg.get("start", 0)) + chunk.start
                    end = float(seg.get("end", 0)) + chunk.start
                    rows.append((start, end, text))
            rows.sort(key=lambda row: row[0])
            for start, end, text in rows:
                count += 1
                output.write(f"{count}\n{timestamp(start)} --> {timestamp(max(end, start + .2))}\n{text}\n\n")
            del data, rows
    return count


def write_bilingual_srt(job, target_path: Path, target_key: str = "normal_text") -> int:
    """Write one SRT cue per aligned source/translation segment (English then target)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target_path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in load_manifest(job):
            source_path = job.transcription_dir / f"chunk_{chunk.number:03d}.json"
            translation_path = job.translation_dir / f"chunk_{chunk.number:03d}.json"
            if not source_path.is_file() or not translation_path.is_file():
                continue
            source_segments = json.loads(source_path.read_text(encoding="utf-8")).get("segments", [])
            translated_segments = json.loads(translation_path.read_text(encoding="utf-8")).get("segments", [])
            if len(source_segments) != len(translated_segments):
                raise RuntimeError(
                    f"Bilingual SRT alignment failed for chunk {chunk.number:03d}: "
                    f"{len(source_segments)} source cues but {len(translated_segments)} translated cues."
                )
            rows = []
            for index, (source_segment, translated_segment) in enumerate(zip(source_segments, translated_segments)):
                source_start = float(source_segment.get("start", 0.0))
                source_end = float(source_segment.get("end", source_start))
                translated_start = float(translated_segment.get("start", source_start))
                translated_end = float(translated_segment.get("end", source_end))
                if abs(source_start - translated_start) > 0.05 or abs(source_end - translated_end) > 0.05:
                    raise RuntimeError(
                        f"Bilingual SRT timing mismatch for chunk {chunk.number:03d}, cue {index + 1}: "
                        f"source={source_start:.3f}-{source_end:.3f}, "
                        f"translation={translated_start:.3f}-{translated_end:.3f}."
                    )
                english = str(source_segment.get("text", "")).strip()
                arabic = str(translated_segment.get(target_key) or translated_segment.get("translated_text") or "").strip()
                if not english and not arabic:
                    continue
                start = source_start + chunk.start
                end = source_end + chunk.start
                body = "\n".join(part for part in (f"EN: {english}" if english else "", f"AR: {arabic}" if arabic else "") if part)
                rows.append((start, end, body))
            rows.sort(key=lambda row: row[0])
            for start, end, body in rows:
                count += 1
                output.write(f"{count}\n{timestamp(start)} --> {timestamp(max(end, start + .2))}\n{body}\n\n")
            del source_segments, translated_segments, rows
    return count


def _valid_json_segments(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = payload.get("segments") if isinstance(payload, dict) else None
        return isinstance(segments, list) and all(isinstance(item, dict) for item in segments)
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return False


def _valid_transcription_checkpoint(job, chunks) -> bool:
    for chunk in chunks:
        base = f"chunk_{chunk.number:03d}"
        json_path = job.transcription_dir / f"{base}.json"
        srt_path = job.transcription_dir / f"{base}.srt"
        if not _valid_json_segments(json_path) or not srt_path.is_file():
            return False
        try:
            segments = json.loads(json_path.read_text(encoding="utf-8")).get("segments", [])
            expected_count = job.get_chunk_state(chunk.number).get("segment_count")
            if expected_count is not None and len(segments) != int(expected_count):
                return False
            if segments and srt_path.stat().st_size == 0:
                return False
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
    return bool(chunks)


def _valid_translation_checkpoint(job, chunks) -> bool:
    for chunk in chunks:
        base = f"chunk_{chunk.number:03d}"
        json_path = job.translation_dir / f"{base}.json"
        normal = job.translation_dir / f"{base}_normal.srt"
        fitted = job.translation_dir / f"{base}_fitted.srt"
        if not _valid_json_segments(json_path) or not normal.is_file() or not fitted.is_file():
            return False
        try:
            segments = json.loads(json_path.read_text(encoding="utf-8")).get("segments", [])
            expected_count = job.get_chunk_state(chunk.number).get("translated_segment_count")
            if expected_count is not None and len(segments) != int(expected_count):
                return False
            if segments and (normal.stat().st_size == 0 or fitted.stat().st_size == 0):
                return False
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
    return bool(chunks)


def _valid_split_checkpoint(job, source: Path, duration: float, should_split: bool, chunk_minutes: float, search_window: float):
    manifest_path = job.status_dir / "chunks_manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = load_manifest(job)
        if not chunks or Path(data.get("source", "")).resolve() != source.resolve():
            return False, []
        if abs(float(data.get("duration", 0.0)) - duration) > max(2.0, duration * 0.01):
            return False, []
        if should_split:
            if abs(float(data.get("chunk_minutes", -1)) - chunk_minutes) > 1e-6 or abs(float(data.get("search_window", -1)) - search_window) > 1e-6:
                return False, []
            if any(not is_valid_audio_chunk(chunk.path, chunk.duration) for chunk in chunks):
                return False, []
        elif len(chunks) != 1 or Path(chunks[0].path).resolve() != source.resolve():
            return False, []
        return True, chunks
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError):
        return False, []


def make_single_chunk(job, source: Path, duration: float) -> None:
    manifest = job.status_dir / "chunks_manifest.json"
    job.start_stage("split")
    atomic_write_text(manifest, json.dumps({"source": str(source), "duration": duration, "boundaries": [0.0, duration],
                                           "chunks": [{"number": 1, "start": 0.0, "end": duration, "path": str(source)}]},
                                          ensure_ascii=False, indent=2))
    job.set_source(input_path=str(source), duration=duration, short_clip_no_split=True)
    job.update_state(chunk_count=1, chunks_manifest=str(manifest.resolve()))
    job.set_chunk_state(1, status="split_completed", start=0.0, end=duration, duration=duration, chunk_path=str(source))
    job.finish_stage("split")


def write_used_srt_and_report(job, source: Path, final_dir: Path, elapsed: float) -> dict:
    records = []
    for chunk in load_manifest(job):
        path = job.status_dir / f"chunk_{chunk.number:03d}_segment_manifest.json"
        if not path.exists():
            continue
        source_path = job.transcription_dir / f"chunk_{chunk.number:03d}.json"
        source_segments = json.loads(source_path.read_text(encoding="utf-8")).get("segments", []) if source_path.exists() else []
        for row in json.loads(path.read_text(encoding="utf-8")):
            item = dict(row)
            source_index = max(0, int(item.get("segment_index", 1)) - 1)
            item["source_text"] = str(source_segments[source_index].get("text", "")).strip() if source_index < len(source_segments) else ""
            item["start"] = float(item.get("start", 0)) + chunk.start
            item["end"] = float(item.get("end", 0)) + chunk.start
            records.append(item)
    records.sort(key=lambda x: (x["start"], x["chunk"], x["segment_index"]))
    cues = []
    for number, row in enumerate(records, 1):
        text = str(row.get("selected_text", "")).strip()
        if text:
            cues += [str(number), f"{timestamp(row['start'])} --> {timestamp(max(row['end'], row['start'] + .2))}", text, ""]
    (final_dir / f"{source.stem}_arabic.srt").write_text("\n".join(cues), encoding="utf-8")
    bilingual_cues = []
    for number, row in enumerate(records, 1):
        english = str(row.get("source_text", "")).strip()
        arabic = str(row.get("selected_text", "")).strip()
        body = "\n".join(part for part in (f"EN: {english}" if english else "", f"AR: {arabic}" if arabic else "") if part)
        if body:
            bilingual_cues += [str(len(bilingual_cues) // 4 + 1), f"{timestamp(row['start'])} --> {timestamp(max(row['end'], row['start'] + .2))}", body, ""]
    bilingual_srt = final_dir / f"{source.stem}_bilingual_en_ar.srt"
    bilingual_srt.write_text("\n".join(bilingual_cues), encoding="utf-8")
    normal = sum(x.get("selected_mode") == "normal" for x in records)
    fitted = sum(x.get("used_fitted_fallback") for x in records)
    sped = [x for x in records if float(x.get("speed_factor", 1.0)) > 1.001]
    trimmed = [x for x in records if x.get("safety_trim_applied")]
    overlaps = [x for x in records if float(x.get("overlap_seconds", 0.0)) > .05]
    report = {
        "input": str(source), "workspace": str(job.root), "segments": len(records),
        "normal_segments": normal, "fitted_segments": fitted, "sped_up_segments": len(sped),
        "highest_speed_factor": max([float(x.get("speed_factor", 1.0)) for x in records] or [1.0]),
        "safety_trim_segments": [{"chunk": x["chunk"], "segment_index": x["segment_index"], "start": x["start"], "end": x["end"]} for x in trimmed],
        "overlap_segments": [{"chunk": x["chunk"], "segment_index": x["segment_index"], "overlap_seconds": x["overlap_seconds"]} for x in overlaps],
        "elapsed_seconds": round(elapsed, 2), "segment_manifest": str(job.status_dir / "segment_manifest.json"),
        "bilingual_srt": str(bilingual_srt), "bilingual_cues": len(bilingual_cues) // 4,
    }
    (job.status_dir / "segment_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (final_dir / "dubbing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["Dubbing report", f"Input: {source}", f"Segments: {len(records)}", f"Normal: {normal}", f"Fitted fallback: {fitted}",
             f"Speeded up: {len(sped)}", f"Highest speed: {report['highest_speed_factor']:.3f}x", f"Safety trims: {len(trimmed)}", f"Overlaps: {len(overlaps)}", f"Elapsed: {elapsed:.2f}s"]
    (final_dir / "dubbing_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Local-file dubbing pipeline")
    p.add_argument("input", help="Local input video or audio path")
    p.add_argument("--source-language", default="auto", choices=list(SOURCE_MAP))
    p.add_argument("--target-language", default="Arabic"); p.add_argument("--target-code", default="ar")
    p.add_argument("--llm", default="Fanar-1-9B-Instruct (Arabic/English — 8 GB+ VRAM)")
    p.add_argument("--whisper", default="large-v3"); p.add_argument("--whisper-batch-size", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gender", default="Man", choices=["Man", "Woman"]); p.add_argument("--reference-audio")
    p.add_argument("--external-srt", help="Use this existing SRT as the dubbing text and timings")
    p.add_argument("--output", default="video", choices=["video", "audio"])
    p.add_argument("--task", choices=["full", "split", "transcribe", "translate", "translate_srt", "bilingual_srt", "dub", "srt_dub"], default="full")
    p.add_argument("--text-mode", choices=["auto", "normal", "fitted"], default="normal")
    p.add_argument("--split-mode", choices=["auto", "never", "always"], default="auto")
    p.add_argument("--chunk-minutes", type=float, default=10.0); p.add_argument("--split-threshold-seconds", type=float, default=1140.0)
    p.add_argument("--search-window", type=float, default=30.0); p.add_argument("--max-speed", type=float, default=2.5)
    p.add_argument("--redub", action="store_true")
    args = p.parse_args()
    run_started = time.perf_counter()

    if urlparse(args.input).scheme.lower() in {"http", "https"}:
        raise SystemExit("ERROR: HTTP/HTTPS URLs are not supported. Provide a local file path.")
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"ERROR: Local input file not found: {source}")
    if args.reference_audio and (urlparse(args.reference_audio).scheme.lower() in {"http", "https"} or not Path(args.reference_audio).expanduser().is_file()):
        raise SystemExit("ERROR: Reference audio must be an existing local file.")
    if args.external_srt and (urlparse(args.external_srt).scheme.lower() in {"http", "https"} or not Path(args.external_srt).expanduser().is_file()):
        raise SystemExit("ERROR: External SRT must be an existing local file.")
    if args.task == "srt_dub" and not args.external_srt:
        raise SystemExit("ERROR: --task srt_dub requires --external-srt.")
    if not 1.0 <= args.max_speed <= 2.5:
        raise SystemExit("ERROR: --max-speed must be between 1.0 and 2.5.")
    os.environ["ZAST_DEVICE"] = args.device
    os.environ["ZAST_TTS_DEVICE"] = args.device
    import config as app_config
    if args.device == "cuda" and not app_config.torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA was requested but torch.cuda.is_available() is False.")
    app_config.DEVICE = args.device
    for module_name in ("long_video_pipeline.transcription_stage", "long_video_pipeline.dubbing_stage",
                        "modules.separator", "modules.translator", "modules.llm_backends.local_causal",
                        "modules.llm_backends.llama_cpp_backend", "modules.tts_backends.voxcpm2"):
        module = __import__(module_name, fromlist=["DEVICE"])
        if hasattr(module, "DEVICE"):
            module.DEVICE = args.device
    time_sync_module.MAX_SPEED_FACTOR = args.max_speed

    workspace = Path(os.environ.get("ZAST_WORK_DIR", str(source.parent / f"{source.stem}_dubbing"))).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("status", "config", "chunks", "transcription", "translation", "separated", "dubbing", "temp", "logs", "final"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    state_file = workspace / "status" / "job_state.json"
    job = JobContext(workspace.parent, job_id=workspace.name, create=not state_file.exists())
    job.set_settings(source=str(source), target_language=args.target_language, target_code=args.target_code,
                     llm=args.llm, whisper=args.whisper, device=args.device, gender=args.gender,
                     output=args.output, reference_audio=args.reference_audio, max_speed=args.max_speed,
                     task=args.task, text_mode=args.text_mode, whisper_batch_size=args.whisper_batch_size,
                     split_mode=args.split_mode, split_threshold_seconds=args.split_threshold_seconds,
                     chunk_minutes=args.chunk_minutes, search_window=args.search_window)
    (job.config_dir / "settings.json").write_text(json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    duration = probe_duration(source)
    job.set_source(input_path=str(source), duration=duration, source_key=str(source).casefold())
    split_path = job.status_dir / "chunks_manifest.json"

    if args.redub:
        for directory in (job.dubbing_dir, job.final_dir):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        state = job.load_state()
        for stage in ("dub", "assemble", "package"):
            if stage in state.get("stages", {}):
                state["stages"][stage].update(status="pending", started_at=None, finished_at=None, error=None)
        for item in state.get("chunks", {}).values():
            for key in list(item):
                if key.startswith("dubbing") or key in {"dubbed_voice", "mixed_audio", "sync_stats"}:
                    item.pop(key, None)
        job.save_state(state)

    should_split = args.split_mode == "always" or (args.split_mode == "auto" and duration >= args.split_threshold_seconds)
    if args.split_mode == "never":
        should_split = False
    split_valid, chunks = _valid_split_checkpoint(job, source, duration, should_split, args.chunk_minutes, args.search_window)
    if not split_valid:
        if split_path.exists():
            job.log("[RESUME] Split checkpoint/manifest is incomplete or invalid; rebuilding only invalid split artifacts.")
        if should_split:
            split_video(source, job, chunk_minutes=args.chunk_minutes, search_window=args.search_window)
        else:
            make_single_chunk(job, source, duration)
        chunks = load_manifest(job)
    elif not job.is_stage_completed("split"):
        job.log("[RESUME] Valid split manifest and chunks found; marking split complete without re-extraction.")
        job.finish_stage("split")
    if not chunks or any(not Path(c.path).is_file() or Path(c.path).stat().st_size <= 0 for c in chunks):
        raise RuntimeError("Split stage is marked complete but one or more chunk files are missing or empty.")
    if args.task == "split":
        job.update_state(status="partial", current_stage=None, last_task="split")
        print(f"Split complete: {len(chunks)} chunk(s)")
        for chunk in chunks:
            print(f"chunk_{chunk.number:03d}: {chunk.start:.3f}s - {chunk.end:.3f}s | {chunk.path}")
        return 0

    lang = SOURCE_MAP[args.source_language]
    if args.task == "srt_dub":
        cue_count = seed_external_srt(job, Path(args.external_srt).expanduser().resolve(), chunks)
        print(f"External SRT loaded: {cue_count} cue(s)")
        transcription_valid = True
        translation_valid = True
    else:
        transcription_valid = _valid_transcription_checkpoint(job, chunks)
        translation_valid = _valid_translation_checkpoint(job, chunks)
    if job.is_stage_completed("transcribe") and not transcription_valid:
        state = job.load_state(); state["stages"]["transcribe"].update(status="pending", finished_at=None, error=None); job.save_state(state)
    if args.task in {"full", "transcribe"}:
        if not job.is_stage_completed("transcribe") or not transcription_valid:
            run_transcription_stage(job, model_size=args.whisper, batch_size=args.whisper_batch_size, language=lang, initial_prompt=source.stem)
    elif not transcription_valid:
        raise SystemExit("Transcription files are missing. Run --task transcribe first.")
    effective_source_lang = lang
    if effective_source_lang is None:
        detected = []
        for chunk in chunks:
            item = json.loads((job.transcription_dir / f"chunk_{chunk.number:03d}.json").read_text(encoding="utf-8"))
            code = str(item.get("language", "unknown")).lower()
            if code not in {"", "unknown", "none"}:
                detected.append(code)
        if detected:
            effective_source_lang = max(set(detected), key=detected.count)
        else:
            effective_source_lang = "en"
            print("[WARN] WhisperX did not provide a detected source language; translation will use en.")
        print(f"[LANGUAGE] Detected source language: {effective_source_lang}")
        job.set_settings(detected_source_language=effective_source_lang)

    if args.task != "srt_dub":
        translation_valid = _valid_translation_checkpoint(job, chunks)
    if args.task == "transcribe":
        cues = write_full_srt(job, job.final_dir / f"{source.stem}_source.srt", "text")
        if cues <= 0:
            raise RuntimeError("Transcription completed but source SRT contains no cues.")
        job.update_state(status="partial", current_stage=None, last_task="transcribe")
        print(f"Transcription ready: {job.final_dir / (source.stem + '_source.srt')}")
        return 0

    if args.task in {"full", "translate", "translate_srt", "bilingual_srt"}:
        if job.is_stage_completed("translate") and not translation_valid:
            state = job.load_state(); state["stages"]["translate"].update(status="pending", finished_at=None, error=None); job.save_state(state)
        if not job.is_stage_completed("translate") or not translation_valid:
            run_translation_stage(job, backend_name=args.llm, source_lang=effective_source_lang or "en", target_lang_name=args.target_language,
                                  target_lang_code=args.target_code, speed_factor=args.max_speed)
    elif not translation_valid:
        raise SystemExit("Translation files are missing. Run --task translate first.")

    final = job.final_dir
    srt_counts = {
        "source": write_full_srt(job, final / f"{source.stem}_source.srt", "text"),
        "normal": write_full_srt(job, final / f"{source.stem}_normal.srt", "normal_text"),
        "fitted": write_full_srt(job, final / f"{source.stem}_fitted.srt", "fitted_text"),
    }
    bilingual_key = "fitted_text" if args.text_mode == "fitted" else "normal_text"
    srt_counts["bilingual_en_ar"] = write_bilingual_srt(job, final / f"{source.stem}_bilingual_en_ar.srt", bilingual_key)
    if any(count <= 0 for count in srt_counts.values()):
        raise RuntimeError(f"SRT validation failed; cue counts: {srt_counts}")
    if args.task in {"translate", "translate_srt", "bilingual_srt"}:
        selected_key = "fitted_text" if args.text_mode == "fitted" else "normal_text"
        selected_cues = write_full_srt(job, final / f"{source.stem}_arabic.srt", selected_key)
        if selected_cues <= 0:
            raise RuntimeError(f"Selected {args.text_mode} translation contains no SRT cues.")
        job.update_state(status="partial", current_stage=None, last_task="translate")
        print(f"Translation ready. Selected text: {args.text_mode if args.text_mode != 'auto' else 'normal'}")
        print(f"Selected Arabic SRT: {final / (source.stem + '_arabic.srt')}")
        print(f"Bilingual EN/AR SRT: {final / (source.stem + '_bilingual_en_ar.srt')}")
        print(f"Normal SRT: {final / (source.stem + '_normal.srt')}")
        print(f"Fitted SRT: {final / (source.stem + '_fitted.srt')}")
        return 0

    results = run_dubbing_stage(job, reference_audio=str(Path(args.reference_audio).resolve()) if args.reference_audio else None,
                                language=args.target_language, gender=args.gender, speed=1.0,
                                voice_volume_db=0.0, background_volume_db=-6.0, tts_device=args.device,
                                text_mode=args.text_mode, max_speed=args.max_speed)
    mixed = [Path(x["mixed_audio"]).resolve() for x in results]
    if not mixed:
        raise RuntimeError("No dubbed audio chunks were produced.")
    concat = final / "audio_concat.txt"
    concat.write_text("\n".join("file '" + str(x).replace("'", "'\\''").replace("\\", "/") + "'" for x in mixed) + "\n", encoding="utf-8")
    final_audio = final / f"{source.stem}_dubbed.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c:a", "aac", "-b:a", "192k", str(final_audio)], check=True)
    if args.output == "video" and source.suffix.lower() not in AUDIO_EXT:
        final_video = final / f"{source.stem}_dubbed.mp4"
        VideoAssembler().assemble(str(source), str(final_audio), str(final_video))
    report = write_used_srt_and_report(job, source, final, time.perf_counter() - run_started)
    if report["segments"] <= 0 or (final / f"{source.stem}_arabic.srt").stat().st_size <= 0:
        raise RuntimeError("Arabic/used SRT validation failed: no spoken segment text was recorded.")
    expected_video = final / f"{source.stem}_dubbed.mp4"
    actual_output = expected_video if args.output == "video" and source.suffix.lower() not in AUDIO_EXT else final_audio
    if not actual_output.is_file() or actual_output.stat().st_size <= 0:
        raise RuntimeError(f"Final output is missing or empty: {actual_output}")
    output_duration = probe_duration(actual_output)
    allowed_delta = max(2.0, duration * 0.01)
    if abs(output_duration - duration) > allowed_delta:
        raise RuntimeError(f"Final duration validation failed: source={duration:.3f}s output={output_duration:.3f}s tolerance={allowed_delta:.3f}s")
    report.update({"source_duration_seconds": duration, "output_duration_seconds": output_duration,
                   "output_duration_delta_seconds": round(output_duration - duration, 3),
                   "validated_output": str(actual_output), "output_size_bytes": actual_output.stat().st_size,
                   "srt_cue_counts": srt_counts,
                   "settings": {"source_language": args.source_language, "detected_source_language": effective_source_lang,
                                "target_language": args.target_language, "target_code": args.target_code,
                                "llm": args.llm, "whisper": args.whisper, "device": args.device,
                                "gender": args.gender, "output": args.output, "max_speed": args.max_speed,
                                "text_mode": args.text_mode, "task": args.task, "whisper_batch_size": args.whisper_batch_size,
                                "reference_audio": args.reference_audio}})
    (final / "dubbing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    job.mark_completed()
    temporary_audio_dirs = [job.temp_dir, *job.dubbing_dir.glob("chunk_*/timesync_temp")]
    for temporary_dir in temporary_audio_dirs:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)
    print("[CLEANUP] Removed temporary Demucs/TTS cache files; source chunks, transcripts, translations, manifests, and final outputs were kept.")
    print(f"Completed audio: {final_audio}")
    if args.output == "video" and source.suffix.lower() not in AUDIO_EXT:
        print(f"Completed video: {final / (source.stem + '_dubbed.mp4')}")
    print(f"Final folder: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
