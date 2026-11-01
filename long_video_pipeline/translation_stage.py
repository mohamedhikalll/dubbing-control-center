from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from config import MAX_SPEED_FACTOR
from fitted_cps_config import get_effective_cps
from modules.reformulator import Reformulator
from modules.srt_parser import SRTParser

from .job_manager import JobContext, atomic_write_text
from .splitter import ChunkInfo, load_manifest


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    segments = data.get("segments", [])

    if not isinstance(segments, list):
        raise ValueError("ملف التفريغ لا يحتوي على قائمة segments صحيحة.")

    normalized: list[dict[str, Any]] = []

    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", "")).strip()

        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start + 0.4))
        except (TypeError, ValueError):
            continue

        if end <= start:
            end = start + 0.4

        item = dict(segment)
        item["index"] = index
        item["start"] = start
        item["end"] = end
        item["text"] = text
        normalized.append(item)

    return normalized


class BatchTranslator:
    """
    تحميل Reformulator مرة واحدة وترجمة كل أجزاء Job بالتتابع.

    ينتج لكل جزء:
        translation/chunk_XXX_normal.srt
        translation/chunk_XXX_fitted.srt
        translation/chunk_XXX.json
    """

    def __init__(
        self,
        backend_name: str = "Qwen3.5-9B",
        source_lang: str = "en",
        target_lang_name: str = "Arabic",
        target_lang_code: str = "ar",
        cps: float | None = None,
        speed_factor: float = MAX_SPEED_FACTOR,
    ):
        self.backend_name = backend_name
        self.source_lang = source_lang
        self.target_lang_name = target_lang_name
        self.target_lang_code = target_lang_code
        self.speed_factor = speed_factor

        self.cps = (
            float(cps)
            if cps is not None
            else float(get_effective_cps(target_lang_code, {}))
        )

        self.reformulator = Reformulator(
            backend_name=self.backend_name,
        )
        self.srt_parser = SRTParser()
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        print(
            f"[TRANSLATE] Loading LLM once: "
            f"{self.backend_name}"
        )
        self.reformulator.load_model()
        self.loaded = True

    def translate_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.load()

        fitted_segments = [dict(segment) for segment in segments]

        print(
            f"[TRANSLATE] Fitted translation: "
            f"{len(fitted_segments)} segments, "
            f"CPS={self.cps}, speed_factor={self.speed_factor}"
        )

        fitted_segments = self.reformulator.translate_segments(
            fitted_segments,
            self.source_lang,
            self.target_lang_name,
            self.target_lang_code,
            cps=self.cps,
            speed_factor=self.speed_factor,
        )

        # نفس منطق التطبيق الحالي:
        # الاختصار الإضافي يطبق فقط على المقاطع التي ما زالت طويلة.
        long_items: list[tuple[int, str, int, dict[str, Any]]] = []

        for index, segment in enumerate(fitted_segments):
            text = str(segment.get("translated_text", "")).strip()
            duration = float(segment["end"]) - float(segment["start"])
            max_chars = max(
                1,
                int(duration * self.cps * self.speed_factor),
            )

            if text and len(text) > max_chars * 1.1:
                long_items.append(
                    (index, text, max_chars, segment)
                )

        if long_items:
            print(
                f"[TRANSLATE] Shortening "
                f"{len(long_items)} overflowing segments"
            )

            items = [
                (text, max_chars)
                for _index, text, max_chars, _segment in long_items
            ]

            shortened_results = self.reformulator.shorten_batch(
                items,
                self.target_lang_code,
                batch_size=8,
            )

            for (
                (index, original_text, _max_chars, segment),
                shortened,
            ) in zip(long_items, shortened_results):
                shortened = str(shortened or "").strip()

                if shortened and len(shortened) < len(original_text):
                    segment["translated_text"] = shortened
                    segment["reformulated"] = True
                    fitted_segments[index] = segment

        print(
            "[TRANSLATE] Natural full translation"
        )

        # نترجم نسخة مستقلة حتى لا تؤثر normal_text على fitted_text.
        translated_segments = [
            dict(segment)
            for segment in fitted_segments
        ]

        translated_segments = self.reformulator.translate_normal(
            translated_segments,
            self.source_lang,
            self.target_lang_code,
        )

        # نعيد fitted_text إلى نفس السجل النهائي.
        for fitted, translated in zip(
            fitted_segments,
            translated_segments,
        ):
            translated["fitted_text"] = fitted.get(
                "translated_text",
                "",
            )
            translated["normal_text"] = translated.get(
                "normal_text",
                translated.get("fitted_text", ""),
            )

        return translated_segments

    def cleanup(self) -> None:
        if not self.loaded:
            return

        print("[TRANSLATE] Cleaning LLM resources")

        try:
            self.reformulator.cleanup()
        finally:
            self.loaded = False
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        print("[TRANSLATE] LLM resources released")


def _write_translation_srts(
    parser: SRTParser,
    segments: list[dict[str, Any]],
    normal_path: Path,
    fitted_path: Path,
    target_lang_code: str,
) -> None:
    normal_path.parent.mkdir(parents=True, exist_ok=True)

    normal_temp = normal_path.with_name(f".{normal_path.stem}.writing{normal_path.suffix}")
    fitted_temp = fitted_path.with_name(f".{fitted_path.stem}.writing{fitted_path.suffix}")
    normal_temp.unlink(missing_ok=True)
    fitted_temp.unlink(missing_ok=True)
    try:
        parser.segments_to_clean_srt(segments, str(normal_temp), text_key="normal_text", lang_code=target_lang_code, clean_fillers=False)
        parser.segments_to_clean_srt(segments, str(fitted_temp), text_key="fitted_text", lang_code=target_lang_code, clean_fillers=False)
        if not normal_temp.is_file() or not fitted_temp.is_file():
            raise RuntimeError("Translation SRT checkpoint was not fully written.")
        if segments and (normal_temp.stat().st_size == 0 or fitted_temp.stat().st_size == 0):
            raise RuntimeError("Translation SRT checkpoint is empty despite translated segments.")
        os.replace(normal_temp, normal_path)
        os.replace(fitted_temp, fitted_path)
    finally:
        normal_temp.unlink(missing_ok=True)
        fitted_temp.unlink(missing_ok=True)


def _valid_translation_artifacts(json_path: Path, normal_srt: Path, fitted_srt: Path, expected_count: int | None = None) -> bool:
    try:
        payload = _read_json(json_path)
        segments = payload.get("segments") if isinstance(payload, dict) else None
        if not isinstance(segments, list) or not all(isinstance(item, dict) for item in segments):
            return False
        if expected_count is not None and len(segments) != int(expected_count):
            return False
        if not normal_srt.is_file() or not fitted_srt.is_file():
            return False
        return not segments or (normal_srt.stat().st_size > 0 and fitted_srt.stat().st_size > 0)
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return False


def run_translation_stage(
    job: JobContext,
    *,
    backend_name: str = "Qwen3.5-9B",
    source_lang: str = "en",
    target_lang_name: str = "Arabic",
    target_lang_code: str = "ar",
    cps: float | None = None,
    speed_factor: float = MAX_SPEED_FACTOR,
) -> list[dict[str, Any]]:
    """
    ترجمة كل ملفات JSON الناتجة من مرحلة التفريغ.

    النتائج تحفظ على القرص فور انتهاء كل جزء.
    إذا توقف البرنامج، يعاد تشغيل الجزء غير المكتمل فقط.
    """
    manifest: list[ChunkInfo] = load_manifest(job)

    if not manifest:
        raise RuntimeError("لا توجد أجزاء في manifest.")

    job.start_stage("translate")

    translator = BatchTranslator(
        backend_name=backend_name,
        source_lang=source_lang,
        target_lang_name=target_lang_name,
        target_lang_code=target_lang_code,
        cps=cps,
        speed_factor=speed_factor,
    )

    results: list[dict[str, Any]] = []

    try:
        for chunk in manifest:
            transcription_json = job.path_for_chunk(
                job.transcription_dir,
                chunk.number,
                ".json",
            )

            translation_json = job.path_for_chunk(
                job.translation_dir,
                chunk.number,
                ".json",
            )
            normal_srt = (
                job.translation_dir
                / f"chunk_{chunk.number:03d}_normal.srt"
            )
            fitted_srt = (
                job.translation_dir
                / f"chunk_{chunk.number:03d}_fitted.srt"
            )

            chunk_state = job.get_chunk_state(chunk.number)

            already_done = (
                chunk_state.get("translation_status") == "completed"
                and _valid_translation_artifacts(translation_json, normal_srt, fitted_srt, chunk_state.get("translated_segment_count"))
            )

            if already_done:
                print(
                    f"[TRANSLATE] Reusing "
                    f"chunk_{chunk.number:03d}"
                )
                results.append({"chunk_number": chunk.number, "segment_count": int(chunk_state.get("translated_segment_count", 0)), "reused": True})
                continue

            if not transcription_json.exists():
                raise FileNotFoundError(
                    f"ملف تفريغ الجزء غير موجود: "
                    f"{transcription_json}"
                )

            job.set_chunk_state(
                chunk.number,
                status="translating",
                translation_status="running",
                translation_json=str(
                    translation_json.resolve()
                ),
                normal_srt=str(normal_srt.resolve()),
                fitted_srt=str(fitted_srt.resolve()),
            )

            try:
                transcription = _read_json(
                    transcription_json
                )
                source_segments = _safe_segments(
                    transcription
                )

                translated_segments = translator.translate_segments(
                    source_segments
                )

                payload = {
                    "chunk_number": chunk.number,
                    "chunk_start": chunk.start,
                    "chunk_end": chunk.end,
                    "source_language": source_lang,
                    "target_language": target_lang_code,
                    "target_language_name": target_lang_name,
                    "cps": translator.cps,
                    "speed_factor": speed_factor,
                    "segments": translated_segments,
                }

                _write_json(translation_json, payload)
                _write_translation_srts(
                    translator.srt_parser,
                    translated_segments,
                    normal_srt,
                    fitted_srt,
                    target_lang_code,
                )

                job.set_chunk_state(
                    chunk.number,
                    status="completed",
                    translation_status="completed",
                    translation_json=str(
                        translation_json.resolve()
                    ),
                    normal_srt=str(normal_srt.resolve()),
                    fitted_srt=str(fitted_srt.resolve()),
                    translated_segment_count=len(
                        translated_segments
                    ),
                )

                results.append({"chunk_number": chunk.number, "segment_count": len(translated_segments), "reused": False})
                job.log(
                    f"[TRANSLATE] Completed chunk "
                    f"{chunk.number:03d}: "
                    f"{len(translated_segments)} segments"
                )

                # لا نحتفظ بنسخ ضخمة من الجزء السابق.
                del source_segments
                del translated_segments
                del payload
                gc.collect()

            except Exception as exc:
                job.set_chunk_state(
                    chunk.number,
                    status="failed",
                    translation_status="failed",
                    translation_error=str(exc),
                )
                raise

        job.finish_stage("translate")
        return results

    except Exception as exc:
        job.fail_stage("translate", exc)
        raise

    finally:
        # النموذج يفرغ مرة واحدة بعد جميع الأجزاء.
        translator.cleanup()


if __name__ == "__main__":
    print(
        "هذه وحدة مرحلة الترجمة. "
        "سيتم استدعاؤها من مشغل الـPipeline."
    )
