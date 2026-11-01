from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | os.PathLike, content: str, *, encoding: str = "utf-8") -> None:
    """Replace a checkpoint atomically so forced shutdown cannot leave half-written text."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class JobContext:
    """
    مدير عملية مستقلة لمعالجة فيديو طويل.

    كل Job يحفظ ملفاته على القرص، ويمكن استكماله بعد التوقف
    دون إعادة تنفيذ المراحل المكتملة.
    """

    STAGES = (
        "download",
        "split",
        "transcribe",
        "translate",
        "separate",
        "dub",
        "assemble",
        "package",
    )

    def __init__(
        self,
        job_root: str | os.PathLike,
        job_id: str | None = None,
        *,
        create: bool = True,
    ):
        self.jobs_root = Path(job_root).expanduser().resolve()
        self.job_id = job_id or self._new_job_id()
        self.root = self.jobs_root / self.job_id

        self.config_dir = self.root / "config"
        self.temp_dir = self.root / "temp"
        self.source_dir = self.root
        self.chunks_dir = self.root / "chunks"
        self.transcription_dir = self.root / "transcription"
        self.translation_dir = self.root / "translation"
        self.separated_dir = self.root / "separated"
        self.dubbing_dir = self.root / "dubbing"
        self.assembly_dir = self.root / "final"
        self.final_dir = self.root / "final"
        self.status_dir = self.root / "status"
        self.logs_dir = self.root / "logs"

        self.state_file = self.status_dir / "job_state.json"
        self.log_file = self.logs_dir / "job.log"

        if create:
            self.create_directories()
            if not self.state_file.exists():
                self.save_state(
                    {
                        "job_id": self.job_id,
                        "created_at": self._now(),
                        "updated_at": self._now(),
                        "status": "created",
                        "current_stage": None,
                        "source": {},
                        "settings": {},
                        "stages": {
                            stage: {
                                "status": "pending",
                                "started_at": None,
                                "finished_at": None,
                                "error": None,
                            }
                            for stage in self.STAGES
                        },
                        "chunks": {},
                    }
                )

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _new_job_id() -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"job_{timestamp}_{uuid.uuid4().hex[:6]}"

    def create_directories(self) -> None:
        for directory in (
            self.config_dir,
            self.temp_dir,
            self.chunks_dir,
            self.transcription_dir,
            self.translation_dir,
            self.separated_dir,
            self.dubbing_dir,
            self.final_dir,
            self.status_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}

        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read job state file: {self.state_file}"
            ) from exc

    def save_state(self, state: dict[str, Any]) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = self._now()

        # كتابة ذرية: لا نترك ملف حالة نصف مكتوب إذا توقف البرنامج فجأة.
        fd, temporary_name = tempfile.mkstemp(
            prefix="job_state_",
            suffix=".tmp",
            dir=str(self.status_dir),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_name, self.state_file)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def update_state(self, **changes: Any) -> dict[str, Any]:
        state = self.load_state()
        state.update(changes)
        self.save_state(state)
        return state

    def set_settings(self, **settings: Any) -> None:
        state = self.load_state()
        state.setdefault("settings", {}).update(settings)
        self.save_state(state)

    def set_source(self, **source: Any) -> None:
        state = self.load_state()
        state.setdefault("source", {}).update(source)
        self.save_state(state)

    def start_stage(self, stage: str) -> None:
        self._validate_stage(stage)
        state = self.load_state()
        stage_state = state.setdefault("stages", {}).setdefault(stage, {})

        stage_state.update(
            {
                "status": "running",
                "started_at": self._now(),
                "finished_at": None,
                "error": None,
            }
        )

        state["status"] = "running"
        state["current_stage"] = stage
        self.save_state(state)
        self.log(f"[STAGE START] {stage}")

    def finish_stage(self, stage: str) -> None:
        self._validate_stage(stage)
        state = self.load_state()
        stage_state = state.setdefault("stages", {}).setdefault(stage, {})

        stage_state.update(
            {
                "status": "completed",
                "finished_at": self._now(),
                "error": None,
            }
        )

        state["current_stage"] = None
        self.save_state(state)
        self.log(f"[STAGE DONE] {stage}")

    def fail_stage(self, stage: str, error: Exception | str) -> None:
        self._validate_stage(stage)
        state = self.load_state()
        stage_state = state.setdefault("stages", {}).setdefault(stage, {})

        stage_state.update(
            {
                "status": "failed",
                "finished_at": self._now(),
                "error": str(error),
            }
        )

        state["status"] = "failed"
        state["current_stage"] = stage
        self.save_state(state)
        self.log(f"[STAGE FAILED] {stage}: {error}")

    def is_stage_completed(self, stage: str) -> bool:
        self._validate_stage(stage)
        state = self.load_state()
        return (
            state.get("stages", {})
            .get(stage, {})
            .get("status")
            == "completed"
        )

    def set_chunk_state(
        self,
        chunk_number: int,
        *,
        status: str,
        **details: Any,
    ) -> None:
        state = self.load_state()
        chunks = state.setdefault("chunks", {})
        chunk_key = f"{chunk_number:03d}"

        item = chunks.setdefault(chunk_key, {})
        item.update(
            {
                "chunk_number": chunk_number,
                "status": status,
                "updated_at": self._now(),
                **details,
            }
        )

        self.save_state(state)

    def get_chunk_state(self, chunk_number: int) -> dict[str, Any]:
        state = self.load_state()
        return state.get("chunks", {}).get(f"{chunk_number:03d}", {})

    def is_chunk_completed(self, chunk_number: int, output_key: str) -> bool:
        item = self.get_chunk_state(chunk_number)
        output_path = item.get(output_key)

        return (
            item.get("status") == "completed"
            and bool(output_path)
            and Path(output_path).exists()
        )

    def path_for_chunk(
        self,
        directory: Path,
        chunk_number: int,
        suffix: str,
    ) -> Path:
        return directory / f"chunk_{chunk_number:03d}{suffix}"

    def log(self, message: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        line = f"{self._now()} {message}\n"

        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(line)

        print(line, end="", flush=True)

    def mark_completed(self) -> None:
        self.update_state(
            status="completed",
            current_stage=None,
            completed_at=self._now(),
        )
        self.log("[JOB DONE]")

    def mark_failed(self, error: Exception | str) -> None:
        self.update_state(
            status="failed",
            current_stage=self.load_state().get("current_stage"),
            error=str(error),
        )
        self.log(f"[JOB FAILED] {error}")

    def _validate_stage(self, stage: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(
                f"Unknown stage '{stage}'. "
                f"Expected one of: {', '.join(self.STAGES)}"
            )

    @classmethod
    def open_existing(
        cls,
        jobs_root: str | os.PathLike,
        job_id: str,
    ) -> "JobContext":
        context = cls(jobs_root, job_id=job_id, create=False)

        if not context.state_file.exists():
            raise FileNotFoundError(
                f"Job state not found: {context.state_file}"
            )

        context.create_directories()
        return context



def _job_slug(value: str, fallback: str = "clip") -> str:
    """Create a Windows-safe, readable job folder name."""
    cleaned = safe_filename(value, fallback=fallback)
    return cleaned[:100].strip(" ._") or fallback


def find_or_create_job(
    output_dir: str | os.PathLike,
    *,
    display_name: str,
    source_key: str,
) -> JobContext:
    """Reuse the previous job for the same source, otherwise create a named job.

    The source_key is stored in job_state.json. This makes resume reliable even
    when the process is stopped between stages or while a chunk is running.
    """
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Prefer an existing incomplete job with the exact source key.
    for state_file in root.glob("*/status/job_state.json"):
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("source", {}).get("source_key") == source_key:
            job_id = state.get("job_id") or state_file.parent.parent.name
            job = JobContext(root, job_id=job_id, create=False)
            job.create_directories()
            job.log("[RESUME] Reusing existing job for the same source")
            return job

    slug = _job_slug(display_name, fallback="clip")
    digest = hashlib.sha1(source_key.encode("utf-8", errors="replace")).hexdigest()[:8]
    job_id = f"{slug}__{digest}"
    return JobContext(root, job_id=job_id, create=True)


def create_job(output_dir: str | os.PathLike) -> JobContext:
    """
    إنشاء Job جديد داخل output_dir.
    """
    return JobContext(output_dir)


def safe_filename(name: str, fallback: str = "file") -> str:
    """
    تنظيف اسم ملف قادم من رابط أو من المستخدم.
    """
    cleaned = re.sub(r"[^\w.\- ]+", "_", str(name), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")

    return cleaned[:180] or fallback


if __name__ == "__main__":
    # اختبار بسيط للملف دون تحميل أي نموذج أو مكتبة خارجية.
    root = Path(__file__).resolve().parent.parent / "output" / "long_jobs"
    job = create_job(root)

    job.set_settings(
        processing_mode="long_video",
        chunk_minutes=10,
        split_at_sentence=True,
    )
    job.log("Job manager test completed")
    print(f"JOB_CREATED={job.root}")
