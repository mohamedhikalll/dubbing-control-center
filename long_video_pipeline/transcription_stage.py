from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import whisperx

from config import DEVICE
from modules.srt_cleaner import (
    apply_asr_corrections_cross_cues,
    fix_inter_cue_casing,
    normalize_timecodes,
    remove_empty_cues_and_redistribute,
)
from modules.srt_parser import merge_sentence_fragments
from modules.transcriber import (
    merge_orphan_punctuation_segments,
    split_long_segments,
)
from modules.utils import cleanup_model

from .job_manager import JobContext, atomic_write_text
from .splitter import ChunkInfo, load_manifest


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))

    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = float(segment.get("start", 0.0))
        end = max(float(segment.get("end", start + 0.4)), start + 0.4)

        lines.extend(
            [
                str(index),
                f"{_timestamp(start)} --> {_timestamp(end)}",
                text,
                "",
            ]
        )

    atomic_write_text(path, "\n".join(lines))


class BatchTranscriber:
    """
    تفريغ جميع الأجزاء باستخدام نموذج WhisperX واحد.

    التسلسل:
        load()
        transcribe_chunk(...) عدة مرات
        cleanup() مرة واحدة
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str | None = None,
        batch_size: int = 4,
        language: str | None = None,
        initial_prompt: str | None = None,
    ):
        self.model_size = model_size
        self.compute_type = (
            compute_type
            if compute_type
            else ("float16" if DEVICE == "cuda" else "int8")
        )
        self.batch_size = batch_size
        self.language = language
        self.initial_prompt = initial_prompt
        self.device = DEVICE

        self.asr_model = None
        self.align_models: dict[str, tuple[Any, Any]] = {}

    def load(self) -> None:
        if self.asr_model is not None:
            return

        print(
            f"[TRANSCRIBE] Loading WhisperX once: "
            f"{self.model_size} on {self.device}, "
            f"compute_type={self.compute_type}"
        )

        asr_options = {
            "condition_on_previous_text": False,
        }

        if self.initial_prompt:
            asr_options["initial_prompt"] = self.initial_prompt

        self.asr_model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
            asr_options=asr_options,
        )

    def _get_alignment_model(self, language_code: str):
        if language_code not in self.align_models:
            print(
                f"[TRANSCRIBE] Loading alignment model for "
                f"language={language_code}"
            )

            self.align_models[language_code] = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )

        return self.align_models[language_code]

    def transcribe_chunk(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        if self.asr_model is None:
            self.load()

        audio_path = Path(audio_path)
        print(f"[TRANSCRIBE] Processing {audio_path.name}")

        audio = whisperx.load_audio(str(audio_path))

        result = self.asr_model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language or self.language,
        )

        detected_language = result.get("language", "unknown")
        raw_segments = result.get("segments", [])

        if not raw_segments:
            return {
                "language": detected_language,
                "segments": [],
            }

        model_a, metadata = self._get_alignment_model(detected_language)

        aligned = whisperx.align(
            raw_segments,
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        segments = aligned.get("segments", [])

        # نفس قواعد تنظيف التفريغ الموجودة في المشروع الحالي.
        segments = split_long_segments(
            segments,
            max_duration=9.0,
            max_chars=110,
        )
        segments = merge_orphan_punctuation_segments(segments)
        segments = merge_sentence_fragments(segments)
        segments = remove_empty_cues_and_redistribute(
            segments,
            text_key="text",
        )
        segments = fix_inter_cue_casing(
            segments,
            text_key="text",
            lang_code=detected_language,
        )
        segments = apply_asr_corrections_cross_cues(
            segments,
            text_key="text",
        )
        segments = normalize_timecodes(
            segments,
            min_gap_ms=40,
            min_cue_duration_ms=400,
            text_key="text",
        )

        del audio
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "language": detected_language,
            "segments": segments,
        }

    def cleanup(self) -> None:
        print("[TRANSCRIBE] Cleaning WhisperX resources")

        for language_code, pair in list(self.align_models.items()):
            model_a, _metadata = pair
            try:
                cleanup_model(model_a)
            except Exception as exc:
                print(
                    f"[TRANSCRIBE] Alignment cleanup warning "
                    f"{language_code}: {exc}"
                )

        self.align_models.clear()

        if self.asr_model is not None:
            try:
                cleanup_model(self.asr_model)
            except Exception as exc:
                print(f"[TRANSCRIBE] ASR cleanup warning: {exc}")

        self.asr_model = None

        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("[TRANSCRIBE] WhisperX resources released")


def _valid_transcription_artifacts(json_path: Path, srt_path: Path, expected_count: int | None = None) -> bool:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        segments = payload.get("segments") if isinstance(payload, dict) else None
        if not isinstance(segments, list) or not all(isinstance(item, dict) for item in segments):
            return False
        if expected_count is not None and len(segments) != int(expected_count):
            return False
        if not srt_path.is_file():
            return False
        return not segments or srt_path.stat().st_size > 0
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return False


def run_transcription_stage(
    job: JobContext,
    *,
    model_size: str = "large-v3",
    compute_type: str | None = None,
    batch_size: int = 4,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    تفريغ جميع أجزاء Job.

    كل جزء ينتج:
        transcription/chunk_XXX.srt
        transcription/chunk_XXX.json
    """

    manifest: list[ChunkInfo] = load_manifest(job)

    if not manifest:
        raise RuntimeError("لا توجد أجزاء في manifest.")

    job.start_stage("transcribe")
    transcriber = BatchTranscriber(
        model_size=model_size,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        initial_prompt=initial_prompt,
    )

    results: list[dict[str, Any]] = []

    try:
        for chunk in manifest:
            srt_path = job.path_for_chunk(
                job.transcription_dir,
                chunk.number,
                ".srt",
            )
            json_path = job.path_for_chunk(
                job.transcription_dir,
                chunk.number,
                ".json",
            )

            existing_state = job.get_chunk_state(chunk.number)

            if (
                existing_state.get("transcription_status") == "completed"
                and _valid_transcription_artifacts(json_path, srt_path, existing_state.get("segment_count"))
            ):
                print(
                    f"[TRANSCRIBE] Reusing {srt_path.name}"
                )
                data = json.loads(
                    json_path.read_text(encoding="utf-8")
                )
                results.append({"chunk_number": chunk.number, "language": data.get("language", "unknown"), "segment_count": len(data.get("segments", [])), "reused": True})
                del data
                continue

            job.set_chunk_state(
                chunk.number,
                status="transcribing",
                transcription_status="running",
                transcription_srt=str(srt_path.resolve()),
                transcription_json=str(json_path.resolve()),
            )

            try:
                data = transcriber.transcribe_chunk(
                    chunk.path,
                    language=language,
                )

                payload = {
                    "chunk_number": chunk.number,
                    "chunk_start": chunk.start,
                    "chunk_end": chunk.end,
                    "language": data.get("language", "unknown"),
                    "segments": data.get("segments", []),
                }

                write_srt(payload["segments"], srt_path)
                atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))

                job.set_chunk_state(
                    chunk.number,
                    status="completed",
                    transcription_status="completed",
                    transcription_srt=str(srt_path.resolve()),
                    transcription_json=str(json_path.resolve()),
                    detected_language=payload["language"],
                    segment_count=len(payload["segments"]),
                )

                results.append({"chunk_number": chunk.number, "language": payload["language"], "segment_count": len(payload["segments"]), "reused": False})
                job.log(
                    f"[TRANSCRIBE] Completed chunk "
                    f"{chunk.number:03d}: "
                    f"{len(payload['segments'])} segments"
                )
                del payload, data

            except Exception as exc:
                job.set_chunk_state(
                    chunk.number,
                    status="failed",
                    transcription_status="failed",
                    transcription_error=str(exc),
                )
                raise

        job.finish_stage("transcribe")
        return results

    except Exception as exc:
        job.fail_stage("transcribe", exc)
        raise

    finally:
        # النموذج يُفرغ مرة واحدة بعد جميع الأجزاء.
        transcriber.cleanup()


if __name__ == "__main__":
    print(
        "هذه وحدة مرحلة التفريغ. "
        "سيتم استدعاؤها من مشغل الـPipeline."
    )
