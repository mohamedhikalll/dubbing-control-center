from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .job_manager import JobContext, atomic_write_text


@dataclass
class ChunkInfo:
    number: int
    start: float
    end: float
    path: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _require_tool(tool: str) -> None:
    result = _run([tool, "-version"])
    if result.returncode != 0:
        raise RuntimeError(
            f"لم يتم العثور على {tool}. "
            f"تأكد أن FFmpeg موجود في PATH."
        )


def probe_duration(input_path: str | Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ]
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"فشل ffprobe في قراءة مدة الملف:\n{result.stderr[-2000:]}"
        )

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"تعذر قراءة مدة الملف من ffprobe: {result.stdout!r}"
        ) from exc

    if duration <= 0:
        raise ValueError("مدة الملف غير صحيحة أو تساوي صفرًا.")

    return duration


def is_valid_audio_chunk(path: str | Path, expected_duration: float, *, tolerance: float | None = None) -> bool:
    """Check an existing AAC/WAV chunk is decodable, contains audio, and is not truncated."""
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size < 128:
        return False
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(candidate)])
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
        actual = float(payload.get("format", {}).get("duration", 0.0))
        has_audio = any(item.get("codec_type") == "audio" for item in payload.get("streams", []))
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    allowed = tolerance if tolerance is not None else max(2.0, float(expected_duration) * 0.005)
    return has_audio and actual > 0 and abs(actual - float(expected_duration)) <= allowed


def detect_silences(
    input_path: str | Path,
    *,
    minimum_silence: float = 0.25,
    silence_threshold_db: float = -30.0,
    search_start: float = 0.0,
    search_end: float | None = None,
) -> list[tuple[float, float]]:
    """
    اكتشاف فترات الصمت باستخدام FFmpeg.

    كل عنصر في النتيجة هو:
        (بداية الصمت، نهاية الصمت)

    threshold أقل من -35 dB مناسب كبداية، ويمكن تغييره لاحقًا
    حسب طبيعة التسجيل.
    """
    start = max(0.0, float(search_start))
    command = ["ffmpeg", "-nostdin", "-hide_banner"]
    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(input_path)]
    if search_end is not None:
        search_duration = max(0.0, float(search_end) - start)
        if search_duration <= 0:
            return []
        command += ["-t", f"{search_duration:.3f}"]
    command += [
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={minimum_silence}",
        "-f", "null", "-",
    ]

    result = _run(command)
    log = f"{result.stdout}\n{result.stderr}"

    starts = [
        start + float(value)
        for value in re.findall(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)", log)
    ]
    ends = [
        start + float(value)
        for value in re.findall(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)", log)
    ]

    silences: list[tuple[float, float]] = []

    for index, start in enumerate(starts):
        if index < len(ends):
            end = ends[index]
        else:
            end = start + minimum_silence

        if end > start:
            silences.append((start, end))

    return silences


def _choose_boundary(
    target: float,
    duration: float,
    silences: Iterable[tuple[float, float]],
    *,
    search_window: float = 30.0,
    minimum_chunk_duration: float = 30.0,
) -> float:
    """
    اختيار نقطة قطع قريبة من الهدف.

    الأولوية:
    1. فترة صمت تحتوي على الهدف.
    2. أقرب نهاية صمت حول الهدف.
    3. الهدف نفسه كحل أخير.
    """
    if target >= duration:
        return duration

    candidates: list[tuple[float, float]] = []

    for silence_start, silence_end in silences:
        if silence_start <= target <= silence_end:
            return min(max(target, minimum_chunk_duration), duration)
        midpoint = (silence_start + silence_end) / 2.0

        if abs(midpoint - target) <= search_window:
            candidates.append((abs(midpoint - target), midpoint))

        if abs(silence_start - target) <= search_window:
            candidates.append((abs(silence_start - target), silence_start))

        if abs(silence_end - target) <= search_window:
            candidates.append((abs(silence_end - target), silence_end))

    candidates.sort(key=lambda item: item[0])

    for _, candidate in candidates:
        candidate = max(candidate, minimum_chunk_duration)
        candidate = min(candidate, duration)

        if candidate > 0 and candidate < duration:
            return candidate

    return min(max(target, minimum_chunk_duration), duration)


def calculate_boundaries(
    duration: float,
    silences: list[tuple[float, float]],
    *,
    chunk_minutes: float = 10.0,
    search_window: float = 30.0,
    silence_search=None,
) -> list[float]:
    """
    إنشاء حدود المقاطع.

    مثال:
        [0, 607.2, 1208.6, 1800.0]
    """
    if duration <= 0:
        raise ValueError("مدة الفيديو يجب أن تكون أكبر من صفر.")

    chunk_seconds = max(60.0, chunk_minutes * 60.0)
    boundaries = [0.0]
    current = 0.0

    while current + chunk_seconds < duration:
        target = current + chunk_seconds
        local_start = max(current + 30.0, target - search_window)
        local_end = min(duration, target + search_window)
        if silence_search is not None:
            local_silences = silence_search(local_start, local_end)
        else:
            local_silences = [
                (start, end) for start, end in silences
                if end >= local_start and start <= local_end
            ]
        boundary = _choose_boundary(
            target,
            duration,
            local_silences,
            search_window=search_window,
            minimum_chunk_duration=current + 30.0,
        )

        # منع حلقة لا نهائية أو جزء قصير جدًا.
        if boundary <= current + 30.0:
            boundary = min(target, duration)

        boundaries.append(boundary)
        current = boundary

    if boundaries[-1] < duration:
        boundaries.append(duration)

    return boundaries


def _write_chunk(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> None:
    length = end - start

    if length <= 0.1:
        raise ValueError(f"مدة الجزء غير صحيحة: {start} -> {end}")

    # WhisperX and Demucs only need audio. Extract audio-only chunks so the
    # split stage never re-encodes video; final assembly uses the source video.
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{length:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]

    result = _run(command)

    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"فشل إنشاء الجزء {output_path.name}:\n"
            f"{result.stderr[-3000:]}"
        )


def split_video(
    input_path: str | Path,
    job: JobContext,
    *,
    chunk_minutes: float = 10.0,
    search_window: float = 30.0,
    minimum_silence: float = 0.25,
    silence_threshold_db: float = -30.0,
) -> list[ChunkInfo]:
    """
    إنشاء مقاطع صوتية داخل مجلد الـJob عند فواصل هادئة دقيقة.

    يحتفظ ملف manifest بحدود التوقيت؛ ولا يعاد ترميز الفيديو الوسيط.
    """
    _require_tool("ffmpeg")
    _require_tool("ffprobe")

    input_path = Path(input_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"ملف الإدخال غير موجود: {input_path}")

    job.chunks_dir.mkdir(parents=True, exist_ok=True)
    job.start_stage("split")

    try:
        duration = probe_duration(input_path)
        job.log(f"[SPLIT] Source duration: {duration:.3f}s")

        job.log(f"[SPLIT] Searching for quiet gaps only within ±{search_window:.1f}s of each target.")
        scanned_silences: list[tuple[float, float]] = []

        def search_local_silences(start: float, end: float) -> list[tuple[float, float]]:
            found = detect_silences(
                input_path,
                minimum_silence=minimum_silence,
                silence_threshold_db=silence_threshold_db,
                search_start=start,
                search_end=end,
            )
            scanned_silences.extend(found)
            job.log(f"[SPLIT] Quiet-gap search {start:.1f}-{end:.1f}s: {len(found)} interval(s).")
            return found

        boundaries = calculate_boundaries(
            duration,
            [],
            chunk_minutes=chunk_minutes,
            search_window=search_window,
            silence_search=search_local_silences,
        )
        silences = sorted(set(scanned_silences))
        if not silences:
            job.log("[WARN] No quiet gaps were found near chunk targets; target times will be used and may intersect speech.")

        chunks: list[ChunkInfo] = []

        for number, (start, end) in enumerate(
            zip(boundaries, boundaries[1:]),
            start=1,
        ):
            output_path = job.path_for_chunk(
                job.chunks_dir,
                number,
                ".m4a",
            )

            # Reuse only a decodable full-length chunk. A force-closed FFmpeg may leave a
            # non-empty but truncated target, so extract to a temporary file and promote atomically.
            temporary_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
            if is_valid_audio_chunk(output_path, end - start):
                job.log(f"[SPLIT] Reusing validated {output_path.name}")
            else:
                if output_path.exists():
                    job.log(f"[RESUME] Discarding incomplete/corrupt split checkpoint {output_path.name}")
                    output_path.unlink()
                temporary_path.unlink(missing_ok=True)
                job.log(
                    f"[SPLIT] Extracting audio chunk {number:03d}/{len(boundaries)-1} (AAC, no video re-encode): "
                    f"{start:.3f}s -> {end:.3f}s"
                )
                _write_chunk(input_path, temporary_path, start, end)
                if not is_valid_audio_chunk(temporary_path, end - start):
                    temporary_path.unlink(missing_ok=True)
                    raise RuntimeError(f"Split output failed integrity check: {temporary_path}")
                os.replace(temporary_path, output_path)

            info = ChunkInfo(
                number=number,
                start=start,
                end=end,
                path=str(output_path.resolve()),
            )
            chunks.append(info)

            job.set_chunk_state(
                number,
                status="split_completed",
                start=start,
                end=end,
                duration=info.duration,
                chunk_path=str(output_path.resolve()),
            )

        manifest_path = job.status_dir / "chunks_manifest.json"
        atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "source": str(input_path),
                    "duration": duration,
                    "chunk_minutes": chunk_minutes,
                    "search_window": search_window,
                    "silences": silences,
                    "boundaries": boundaries,
                    "chunks": [asdict(item) for item in chunks],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        job.set_source(
            input_path=str(input_path),
            duration=duration,
        )
        job.update_state(
            chunk_count=len(chunks),
            chunks_manifest=str(manifest_path.resolve()),
        )
        job.finish_stage("split")

        return chunks

    except Exception as exc:
        job.fail_stage("split", exc)
        raise


def load_manifest(job: JobContext) -> list[ChunkInfo]:
    manifest_path = job.status_dir / "chunks_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"ملف تقسيم الأجزاء غير موجود: {manifest_path}"
        )

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        ChunkInfo(**chunk)
        for chunk in data.get("chunks", [])
    ]


if __name__ == "__main__":
    print(
        "هذا الملف وحدة تقسيم فقط. "
        "سيتم استدعاؤه لاحقًا من مشغل الـPipeline."
    )
