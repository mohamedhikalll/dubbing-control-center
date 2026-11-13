from __future__ import annotations

import gc
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from config import DEVICE
from modules.separator import VocalSeparator
from modules.tts_backends.voxcpm2 import VoxCPM2Backend
from modules.time_sync import TimeSync
from modules.audio_mixer import AudioMixer
import modules.time_sync as time_sync_module

from .job_manager import JobContext
from .splitter import ChunkInfo, load_manifest, is_valid_audio_chunk


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _ffmpeg(command: list[str], description: str) -> None:
    result = _run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed during {description}:\n"
            f"{result.stderr[-3000:]}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_duration(path: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot read duration of {path}:\n{result.stderr[-2000:]}"
        )

    return float(result.stdout.strip())


def _move_if_needed(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))

    return target


def _build_silent_voice_track(
    output_path: Path,
    duration: float,
    sample_rate: int = 24000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{max(duration, 0.1):.3f}",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        "creating silent voice track",
    )


def _place_segment_on_timeline(
    segment_audio: Path,
    output_path: Path,
    start: float,
    total_duration: float,
    sample_rate: int = 24000,
) -> None:
    """
    يضع صوت الجملة داخل مسار صامت كامل بطول الجزء.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(segment_audio),
            "-af",
            (
                f"adelay={max(0, int(start * 1000))}:all=1,"
                f""
            ),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        "placing generated speech on timeline",
    )


def _mix_voice_segments(
    segment_tracks: list[tuple[Path, float]],
    output_path: Path,
    duration: float,
    sample_rate: int = 24000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not segment_tracks:
        _build_silent_voice_track(
            output_path,
            duration,
            sample_rate,
        )
        return

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]

    for track, _start in segment_tracks:
        command.extend(["-i", str(track)])

    filters = []

    for index, (_track, start) in enumerate(segment_tracks):
        delay_ms = max(0, int(round(start * 1000)))

        filters.append(
            f"[{index}:a]"
            f"adelay={delay_ms}:all=1,"
            f"atrim=duration={duration:.3f},"
            f"aresample={sample_rate}[a{index}]"
        )

    labels = "".join(
        f"[a{index}]"
        for index in range(len(segment_tracks))
    )

    filters.append(
        f"{labels}"
        f"amix=inputs={len(segment_tracks)}:"
        f"duration=longest:dropout_transition=0,"
        f"atrim=duration={duration:.3f}[aout]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[aout]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )

    _ffmpeg(
        command,
        "mixing generated speech segments",
    )

def _mix_with_background(
    voice_path: Path,
    background_path: Path,
    output_path: Path,
    voice_volume_db: float = 0.0,
    background_volume_db: float = -6.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(background_path),
            "-i",
            str(voice_path),
            "-filter_complex",
            (
                f"[0:a]volume={background_volume_db}dB[bg];"
                f"[1:a]volume={voice_volume_db}dB[voice];"
                "[bg][voice]amix=inputs=2:duration=longest:"
                "dropout_transition=0[aout]"
            ),
            "-map",
            "[aout]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ],
        "mixing dubbed voice with background",
    )



def _retime_audio(
    input_path: Path,
    output_path: Path,
    speed: float,
    max_duration: float,
    sample_rate: int,
) -> None:
    """
    يسرّع الصوت ويقصه عند نهاية زمن الجملة.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    speed = max(1.0, min(float(speed), 2.5))
    remaining = speed
    tempo_filters = []
    while remaining > 2.0:
        tempo_filters.append("atempo=2.0")
        remaining /= 2.0
    tempo_filters.append(f"atempo={remaining:.5f}")

    _ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(input_path),
            "-filter:a",
            (
                f"{','.join(tempo_filters)},"
                f"atrim=duration={max_duration:.3f}"
            ),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        "retiming generated speech",
    )


def _generate_chunk_voice(
    tts: VoxCPM2Backend,
    segments: list[dict[str, Any]],
    chunk_duration: float,
    output_dir: Path,
    *,
    reference_audio: str | None = None,
    language: str = "Arabic",
    gender: str = "Man",
    speed: float = 1.0,
) -> Path:
    """
    إنشاء صوت الدبلجة مع منع تداخل الجمل.

    الخطة:
    1. استخدام normal_text.
    2. قياس مدة الصوت الناتج.
    3. تسريع حتى 2.5x عند الحاجة.
    4. استخدام fitted_text إذا كان normal_text طويلًا جدًا.
    5. قص نهائي عند نهاية زمن الجملة كحل أمان.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    timeline_tracks: list[tuple[Path, float]] = []
    sample_rate = tts._sample_rate or 24000

    for index, segment in enumerate(segments, start=1):
        normal_text = str(
            segment.get("normal_text")
            or segment.get("text")
            or ""
        ).strip()

        fitted_text = str(
            segment.get("fitted_text")
            or segment.get("translated_text")
            or normal_text
        ).strip()

        if not normal_text:
            continue

        start_time = max(
            0.0,
            float(segment.get("start", 0.0)),
        )
        end_time = max(
            start_time + 0.4,
            float(segment.get("end", start_time + 0.4)),
        )
        slot_duration = max(0.4, end_time - start_time)

        normal_audio = (
            segment_dir / f"speech_{index:04d}_normal.wav"
        )
        fitted_audio = (
            segment_dir / f"speech_{index:04d}_fitted.wav"
        )
        final_audio = (
            segment_dir / f"speech_{index:04d}_final.wav"
        )

        if not normal_audio.exists():
            print(
                f"[DUB] TTS normal {index}/{len(segments)}: "
                f"{normal_text[:70]}"
            )

            tts.generate(
                text=normal_text,
                language=language,
                output_path=str(normal_audio),
                ref_audio_path=reference_audio,
                speed=speed,
                duration=slot_duration,
                gender=gender,
            )

        normal_duration = _probe_duration(normal_audio)
        chosen_audio = normal_audio

        if normal_duration > slot_duration + 0.08:
            required_speed = normal_duration / slot_duration

            if required_speed <= 2.5:
                print(
                    f"[DUB] Speed segment {index}: "
                    f"{normal_duration:.2f}s -> "
                    f"{slot_duration:.2f}s "
                    f"({required_speed:.2f}x)"
                )

                _retime_audio(
                    normal_audio,
                    final_audio,
                    required_speed,
                    slot_duration,
                    sample_rate,
                )
                chosen_audio = final_audio

            else:
                print(
                    f"[DUB] Normal text too long for segment {index}; "
                    f"using fitted_text"
                )

                if not fitted_audio.exists():
                    tts.generate(
                        text=fitted_text,
                        language=language,
                        output_path=str(fitted_audio),
                        ref_audio_path=reference_audio,
                        speed=speed,
                        duration=slot_duration,
                        gender=gender,
                    )

                fitted_duration = _probe_duration(fitted_audio)
                fitted_speed = fitted_duration / slot_duration

                if fitted_speed > 1.0:
                    print(
                        f"[DUB] Speed fitted segment {index}: "
                        f"{fitted_duration:.2f}s -> "
                        f"{slot_duration:.2f}s "
                        f"({min(fitted_speed, 2.5):.2f}x)"
                    )

                    _retime_audio(
                        fitted_audio,
                        final_audio,
                        min(fitted_speed, 2.5),
                        slot_duration,
                        sample_rate,
                    )
                    chosen_audio = final_audio
                else:
                    chosen_audio = fitted_audio

        timeline_tracks.append((chosen_audio, start_time))

    voice_path = output_dir / "dubbed_voice.wav"

    if not voice_path.exists():
        _mix_voice_segments(
            timeline_tracks,
            voice_path,
            chunk_duration,
            sample_rate=sample_rate,
        )

    return voice_path


def _valid_dubbing_manifest(path: Path, chunk_number: int) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(rows, list) and bool(rows) and all(
            isinstance(row, dict) and int(row.get("chunk", -1)) == chunk_number
            and int(row.get("segment_index", 0)) > 0 for row in rows
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _separate_chunk(
    separator: VocalSeparator,
    chunk: ChunkInfo,
    job: JobContext,
) -> tuple[Path, Path]:
    """
    فصل صوت الجزء ثم نسخ النتائج إلى مجلد Job مستقل.
    """
    source = Path(chunk.path)

    vocals_target = (
        job.separated_dir
        / f"chunk_{chunk.number:03d}_vocals.wav"
    )
    background_target = (
        job.separated_dir
        / f"chunk_{chunk.number:03d}_background.wav"
    )

    tolerance = max(3.0, float(chunk.duration) * 0.005)
    vocals_valid = is_valid_audio_chunk(vocals_target, chunk.duration, tolerance=tolerance)
    background_valid = is_valid_audio_chunk(background_target, chunk.duration, tolerance=tolerance)
    if vocals_valid and background_valid:
        return vocals_target, background_target
    if vocals_valid and not background_valid:
        # The separator already treats vocals as the safe fallback when no
        # background track is produced. Rebuild only the missing side instead
        # of running the expensive separation again after an interruption.
        if background_target.exists():
            background_target.unlink()
        shutil.copy2(vocals_target, background_target)
        if not is_valid_audio_chunk(background_target, chunk.duration, tolerance=tolerance):
            background_target.unlink(missing_ok=True)
            raise RuntimeError(f"Could not rebuild background checkpoint from {vocals_target.name}")
        job.log(f"[RESUME] Rebuilt missing {background_target.name} from validated vocals checkpoint")
        return vocals_target, background_target

    for candidate, valid in ((vocals_target, vocals_valid), (background_target, background_valid)):
        if candidate.exists() and not valid:
            job.log(f"[RESUME] Removing incomplete separation checkpoint {candidate.name}")
            candidate.unlink()

    print(
        f"[DUB] Separating chunk {chunk.number:03d}"
    )

    separated = separator.separate(str(source))

    vocals_source = Path(separated["vocals"])
    background_source = Path(separated["background"])

    if not vocals_source.exists():
        raise FileNotFoundError(
            f"Vocals file was not created: {vocals_source}"
        )

    vocals_target = _move_if_needed(
        vocals_source,
        vocals_target,
    )

    if background_source.exists():
        background_target = _move_if_needed(
            background_source,
            background_target,
        )
    else:
        # في حالة عدم إنشاء background، نستخدم vocals
        # كبديل حتى لا تفشل العملية.
        shutil.copy2(vocals_target, background_target)

    return vocals_target, background_target


def run_dubbing_stage(
    job: JobContext,
    *,
    reference_audio: str | None = None,
    language: str = "Arabic",
    gender: str = "Man",
    speed: float = 1.0,
    voice_volume_db: float = 0.0,
    background_volume_db: float = -6.0,
    tts_device: str | None = None,
    text_mode: str = "normal",
    max_speed: float = 2.5,
) -> list[dict[str, Any]]:
    """
    Automated dubbing wrapper using the original Dubbing Control Center path:

        TimeSync.sync_all()
        TimeSync.build_full_audio()
        AudioMixer.mix()

    Only the long-video orchestration is automated here.
    """

    import os
    import modules.time_sync as time_sync_module

    manifest: list[ChunkInfo] = load_manifest(job)

    if not manifest:
        raise RuntimeError("No chunks found in manifest.")

    if tts_device:
        os.environ["DUBBING_TTS_DEVICE"] = tts_device
    if text_mode not in {"auto", "normal", "fitted"}:
        raise ValueError("text_mode must be auto, normal, or fitted")
    if not 1.0 <= max_speed <= 2.5:
        raise ValueError("max_speed must be between 1.0 and 2.5")
    time_sync_module.MAX_SPEED_FACTOR = max_speed

    separator = VocalSeparator()
    tts = VoxCPM2Backend()
    mixer = AudioMixer()
    results: list[dict[str, Any]] = []

    try:
        # Separation is retained/reused by _separate_chunk().
        job.start_stage("separate")

        for chunk in manifest:
            vocals_path, background_path = _separate_chunk(
                separator,
                chunk,
                job,
            )

            job.set_chunk_state(
                chunk.number,
                status="separated",
                separation_status="completed",
                vocals_path=str(vocals_path.resolve()),
                background_path=str(background_path.resolve()),
            )

        job.finish_stage("separate")
        job.start_stage("dub")

        tts_loaded = False
        total_chunks = len(manifest)
        desired_dubbing_settings = {
            "text_mode": text_mode,
            "language": language,
            "gender": gender,
            "reference_audio": str(Path(reference_audio).resolve()) if reference_audio else None,
            "device": tts_device,
            "max_speed": max_speed,
        }

        for chunk in manifest:
            print(
                f"[DUB PROGRESS] chunk={chunk.number}/{total_chunks} "
                f"stage=dub stage_percent={int((chunk.number - 1) * 100 / max(1, total_chunks))} "
                f"remaining={max(0, total_chunks - chunk.number + 1)} status=starting",
                flush=True,
            )
            duration = float(chunk.duration)
            chunk_dir = job.dubbing_dir / f"chunk_{chunk.number:03d}"
            voice_path = chunk_dir / "dubbed_voice.wav"
            mixed_path = chunk_dir / "dubbed_mixed.wav"
            state = job.get_chunk_state(chunk.number)
            manifest_path = job.status_dir / f"chunk_{chunk.number:03d}_segment_manifest.json"
            if (
                state.get("dubbing_status") == "completed"
                and is_valid_audio_chunk(voice_path, duration)
                and is_valid_audio_chunk(mixed_path, duration)
                and _valid_dubbing_manifest(manifest_path, chunk.number)
                and state.get("dubbing_settings") == desired_dubbing_settings
            ):
                print(f"[RESUME] Skipping completed dubbing chunk {chunk.number:03d}")
                print(
                    f"[DUB PROGRESS] chunk={chunk.number}/{total_chunks} stage=dub "
                    f"stage_percent={int(chunk.number * 100 / max(1, total_chunks))} "
                    f"remaining={max(0, total_chunks - chunk.number)} status=completed",
                    flush=True,
                )
                results.append({"chunk_number": chunk.number, "mixed_audio": str(mixed_path.resolve())})
                continue

            translation_json = job.path_for_chunk(
                job.translation_dir,
                chunk.number,
                ".json",
            )

            if not translation_json.exists():
                raise FileNotFoundError(
                    f"Translation file missing: {translation_json}"
                )

            data = _read_json(translation_json)
            source_segments = data.get("segments", [])

            if not source_segments:
                raise RuntimeError(
                    f"No translated segments in {translation_json}"
                )

            # Work on a copy so the saved translation JSON is not changed.
            segments = [dict(item) for item in source_segments]

            # Auto preserves normal -> speed -> fitted fallback.
            # Explicit modes select the chosen translated text directly.
            for segment in segments:
                normal = str(
                    segment.get("normal_text")
                    or segment.get("translated_text")
                    or segment.get("text")
                    or ""
                ).strip()
                fitted = str(segment.get("fitted_text") or segment.get("translated_text") or normal).strip()
                segment["translated_text"] = fitted if text_mode == "fitted" else normal

            duration = float(chunk.duration)
            chunk_dir = (
                job.dubbing_dir / f"chunk_{chunk.number:03d}"
            )
            chunk_dir.mkdir(parents=True, exist_ok=True)

            temp_dir = chunk_dir / "timesync_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            voice_path = chunk_dir / "dubbed_voice.wav"
            mixed_path = chunk_dir / "dubbed_mixed.wav"

            state = job.get_chunk_state(chunk.number)
            manifest_path = job.status_dir / f"chunk_{chunk.number:03d}_segment_manifest.json"
            if (
                state.get("dubbing_status") == "completed"
                and is_valid_audio_chunk(voice_path, duration)
                and is_valid_audio_chunk(mixed_path, duration)
                and _valid_dubbing_manifest(manifest_path, chunk.number)
                and state.get("dubbing_settings") == desired_dubbing_settings
            ):
                print(f"[DUB] Reusing chunk {chunk.number:03d}")
                print(
                    f"[DUB PROGRESS] chunk={chunk.number}/{total_chunks} stage=dub "
                    f"stage_percent={int(chunk.number * 100 / max(1, total_chunks))} "
                    f"remaining={max(0, total_chunks - chunk.number)} status=completed",
                    flush=True,
                )
                results.append(
                    {
                        "chunk_number": chunk.number,
                        "mixed_audio": str(mixed_path.resolve()),
                    }
                )
                continue
            if state.get("dubbing_status") == "completed":
                job.log(f"[DUB] Regenerating chunk {chunk.number:03d}: output/settings/manifest is stale")
                for stale in (voice_path, mixed_path):
                    if stale.exists():
                        stale.unlink()
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)

            job.set_chunk_state(
                chunk.number,
                status="dubbing",
                dubbing_status="running",
            )

            # TimeSync uses this module-level directory for its cache.
            # Isolating it prevents chunks from sharing temporary audio.
            old_temp_dir = time_sync_module.TEMP_DIR
            time_sync_module.TEMP_DIR = str(temp_dir)

            try:
                if not tts_loaded:
                    tts.load(ref_audio_path=reference_audio)
                    tts_loaded = True
                syncer = TimeSync(tts_engine=tts, reformulator=None)

                # Use normal text by default; only auto mode may fall back to fitted.
                prepared, _ = syncer.pre_check_and_shorten(
                    segments,
                    language,
                )
                print(
                    f"[DUB] Generating speech for chunk {chunk.number:03d}/{total_chunks} "
                    f"({len(prepared)} segments)",
                    flush=True,
                )

                synced, sync_stats = syncer.sync_all(
                    prepared,
                    language,
                    voice_mapping=None,
                    total_duration=duration,
                    gender=gender,
                )

                fitted_fallbacks: set[int] = set()
                # If a normal sentence still had to be truncated and a
                # fitted version exists, retry that sentence only.
                for index, result in enumerate(synced):
                    if index >= len(prepared):
                        break

                    fitted = str(
                        prepared[index].get("fitted_text")
                        or ""
                    ).strip()
                    normal = str(
                        prepared[index].get("normal_text")
                        or ""
                    ).strip()

                    if (
                        text_mode == "auto"
                        and
                        fitted
                        and fitted != normal
                        and result.get("truncated")
                    ):
                        prepared[index]["translated_text"] = fitted

                        retry = syncer.sync_segment(
                            prepared[index],
                            language,
                            voice_path=reference_audio,
                            gender=gender,
                        )

                        if not retry.get("truncated", False):
                            synced[index] = retry
                            fitted_fallbacks.add(index)
                            print(
                                f"[DUB] Fitted fallback segment "
                                f"{index + 1}"
                            )

                segment_records = []
                for index, segment in enumerate(prepared):
                    result = synced[index] if index < len(synced) else {}
                    normal_text = str(segment.get("normal_text") or segment.get("translated_text") or "").strip()
                    fitted_text = str(segment.get("fitted_text") or "").strip()
                    selected_text = str(result.get("final_text") or segment.get("translated_text") or normal_text).strip()
                    segment_records.append({
                        "chunk": chunk.number,
                        "segment_index": index + 1,
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", 0.0)),
                        "normal_text": normal_text,
                        "fitted_text": fitted_text,
                        "selected_text": selected_text,
                        "selected_mode": "fitted" if text_mode == "fitted" or index in fitted_fallbacks else "normal",
                        "speed_factor": float(result.get("speed_factor", 1.0)),
                        "used_fitted_fallback": index in fitted_fallbacks,
                        "safety_trim_applied": bool(result.get("truncated", False)),
                        "overlap_seconds": float(result.get("overflow", 0.0)),
                    })
                (job.status_dir / f"chunk_{chunk.number:03d}_segment_manifest.json").write_text(
                    json.dumps(segment_records, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                # Original full-track assembly.
                syncer.build_full_audio(
                    synced,
                    duration,
                    output_path=voice_path,
                )

                # Original mixer only. No ducking, compression, or loudnorm.
                mixer.mix(
                    str(voice_path),
                    str(background_path),
                    str(mixed_path),
                    voice_volume_db=voice_volume_db,
                    bg_volume_db=background_volume_db,
                )

                job.set_chunk_state(
                    chunk.number,
                    status="completed",
                    dubbing_status="completed",
                    dubbed_voice=str(voice_path.resolve()),
                    mixed_audio=str(mixed_path.resolve()),
                    sync_stats=sync_stats,
                    dubbing_settings=desired_dubbing_settings,
                )

                results.append(
                    {
                        "chunk_number": chunk.number,
                        "mixed_audio": str(mixed_path.resolve()),
                    }
                )

                job.log(
                    f"[DUB] Completed chunk {chunk.number:03d}"
                )
                print(
                    f"[DUB PROGRESS] chunk={chunk.number}/{total_chunks} stage=dub "
                    f"stage_percent={int(chunk.number * 100 / max(1, total_chunks))} "
                    f"remaining={max(0, total_chunks - chunk.number)} status=completed",
                    flush=True,
                )

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                message = str(exc)
                is_oom = isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)) or "out of memory" in message.lower()
                memory = {}
                if torch.cuda.is_available():
                    try:
                        memory = {
                            "allocated_gib": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
                            "reserved_gib": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
                            "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
                            "gpu": torch.cuda.get_device_name(0),
                        }
                    except Exception:
                        pass
                if is_oom:
                    details = {"stage": "dubbing", "chunk": chunk.number, "exception": message[-4000:], "vram": memory,
                               "next_action": "After this process exits, rerun the same command to resume valid prior stages/chunks, or use --redub to redo dubbing/assembly only. No full-pipeline retry was started."}
                    (job.status_dir / "oom_context.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
                    job.log(f"[OOM] VoxCPM/dubbing chunk {chunk.number:03d}; VRAM={memory}. Cleanup is running. Resume only after this process exits; no automatic retry.")
                job.set_chunk_state(
                    chunk.number,
                    status="failed",
                    dubbing_status="failed",
                    dubbing_error=message,
                    oom=is_oom,
                    vram_at_failure=memory,
                )
                if is_oom and torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                raise

            finally:
                time_sync_module.TEMP_DIR = old_temp_dir

        job.finish_stage("dub")
        return results

    except Exception as exc:
        current_stage = job.load_state().get("current_stage")

        if current_stage in {"separate", "dub"}:
            job.fail_stage(current_stage, exc)

        raise

    finally:
        separator.cleanup()

        try:
            tts.unload()
        except Exception as exc:
            print(f"[DUB] TTS cleanup warning: {exc}")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("[DUB] Original TimeSync/AudioMixer resources released")


if __name__ == "__main__":
    print(
        "هذه وحدة الفصل والدبلجة. "
        "سيتم استدعاؤها من مشغل الـPipeline."
    )
