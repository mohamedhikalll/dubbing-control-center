from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import threading
from modules.process_control import terminate_process_tree, install_console_close_handler, remove_console_close_handler
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse


def _load_tts_factory():
    """Load model dependencies only for an explicit backend-list request."""
    return importlib.import_module("modules.tts_backends.factory")


def get_backend_names() -> list[str]:
    """Read backend names from current and legacy factory implementations."""
    tts_factory_module = _load_tts_factory()
    function = getattr(tts_factory_module, "get_backend_names", None)
    if callable(function):
        return list(function())
    factory = getattr(tts_factory_module, "_factory", None)
    if factory is not None:
        method = getattr(factory, "get_backend_names", None)
        if callable(method):
            return list(method())
        for attr in ("_registry", "_backends"):
            values = getattr(factory, attr, None)
            if isinstance(values, dict) and values:
                return list(values)
    return ["VoxCPM 2"]


def get_backend_catalog(language: str | None = None):
    tts_factory_module = _load_tts_factory()
    function = getattr(tts_factory_module, "get_backend_catalog", None)
    if callable(function):
        return function(language)
    factory = getattr(tts_factory_module, "_factory", None)
    method = getattr(factory, "get_backend_catalog", None) if factory is not None else None
    if callable(method):
        return method(language)
    return None

BASE = Path(__file__).resolve().parent
PROFILE_FILE = BASE / "profiles.json"
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg", ".ts", ".m2ts", ".3gp", ".flv"}
MEDIA_EXT = AUDIO_EXT | VIDEO_EXT
LANGUAGES = ["auto", "en", "ar", "fr", "es", "de", "it", "pt", "ja", "ko", "zh", "ru", "hi"]
VOICES = ["Man", "Woman"]


def _profiles() -> dict:
    try:
        value = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_profile(name: str, values: dict) -> None:
    if not name.strip():
        raise SystemExit("Profile name cannot be empty.")
    data = _profiles()
    data[name] = {k: v for k, v in values.items() if k not in {"input", "profile", "save_profile", "batch_list", "batch_dir", "recursive"}}
    PROFILE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _workspace(source: Path) -> Path:
    return source.parent / f"{source.stem}_dubbing"


def _batch_destination(output_dir: str | os.PathLike, source: str | os.PathLike) -> Path:
    """Give same-named files from different source folders unique export dirs."""
    source_path = Path(source).expanduser().resolve()
    source_key = os.path.normcase(str(source_path)).casefold()
    suffix = hashlib.sha256(source_key.encode("utf-8", errors="replace")).hexdigest()[:8]
    return Path(output_dir).expanduser() / f"{source_path.stem}_{suffix}"


def _is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https"}


def _batch_sources(args: argparse.Namespace) -> list[str]:
    """Resolve a UTF-8 path list or gather supported local media from one folder."""
    if args.batch_list:
        list_path = Path(args.batch_list).expanduser().resolve()
        if not list_path.is_file():
            raise SystemExit(f"Batch list not found: {list_path}")
        sources: list[str] = []
        for line_number, line in enumerate(list_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1].strip()
            value = os.path.expandvars(value)
            if _is_url(value):
                sources.append(value)
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = list_path.parent / path
            sources.append(str(path.resolve()))
    elif args.batch_dir:
        folder = Path(args.batch_dir).expanduser().resolve()
        if not folder.is_dir():
            raise SystemExit(f"Batch folder not found: {folder}")
        candidates = folder.rglob("*") if args.recursive else folder.iterdir()
        sources = []
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXT:
                continue
            relative_parts = path.relative_to(folder).parts
            if any(part.startswith(".") or part.lower().endswith("_dubbing") for part in relative_parts[:-1]):
                continue
            sources.append(str(path.resolve()))
        sources.sort(key=str.casefold)
    else:
        return []

    unique: list[str] = []
    seen: set[str] = set()
    for source in sources:
        key = os.path.normcase(source).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(source)
    if not unique:
        raise SystemExit("No media paths were found in the selected batch source.")
    return unique


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local-file dubbing. URLs are not supported.")
    p.add_argument("input", nargs="?", help="Local video or audio file")
    batch_source = p.add_mutually_exclusive_group()
    batch_source.add_argument("--batch-list", help="UTF-8 TXT file with one local media path per line")
    batch_source.add_argument("--batch-dir", help="Folder containing local video/audio files")
    p.add_argument("--recursive", action="store_true", help="Search subfolders (only with --batch-dir)")
    p.add_argument("--open", "--tui", dest="open_tui", action="store_true", help="Open the interactive control center")
    p.add_argument("--check", action="store_true", help="Check runtime prerequisites and CUDA")
    p.add_argument("--list-languages", action="store_true")
    p.add_argument("--list-voices", action="store_true")
    p.add_argument("--list-tts-backends", action="store_true", help="List registered TTS backends and their availability")
    p.add_argument("--list-profiles", action="store_true")
    p.add_argument("--source-language", choices=LANGUAGES, default="auto")
    p.add_argument("--target-language", default="Arabic")
    p.add_argument("--target-code", default="ar")
    p.add_argument("--llm", default="Fanar-1-9B-Instruct (Arabic/English — 8 GB+ VRAM)")
    p.add_argument("--whisper", default="large-v3")
    p.add_argument("--whisper-batch-size", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--gender", choices=VOICES, default="Man")
    p.add_argument("--reference-audio")
    p.add_argument("--external-srt", help="Use this existing SRT as the dubbing text and timings")
    p.add_argument("--output", choices=["video", "audio"], default="video")
    p.add_argument("--task", choices=["full", "split", "transcribe", "translate", "translate_srt", "bilingual_srt", "dub", "srt_dub"], default="full",
                   help="Run the complete job or one stage only")
    p.add_argument("--text-mode", choices=["auto", "normal", "fitted"], default="normal",
                   help="Text used for synthesis: auto fallback, normal only, or fitted only")
    p.add_argument("--split-mode", choices=["auto", "never", "always"], default="auto")
    p.add_argument("--chunk-minutes", type=float, default=10.0)
    p.add_argument("--split-threshold-seconds", type=float, default=1140.0)
    p.add_argument("--search-window", type=float, default=30.0)
    p.add_argument("--max-speed", type=float, default=2.5,
                   help="Maximum per-sentence speed-up factor (1.0-2.5; only used when a sentence overflows)")
    p.add_argument("--resume", action="store_true", help="Explicitly request reuse of valid completed stages/chunks; automatic by default")
    p.add_argument("--redub", action="store_true", help="Redo dubbing/assembly only")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--profile")
    p.add_argument("--save-profile")
    p.add_argument("--output-dir", help="Copy completed deliverables to this directory")
    p.add_argument("--log-file", help="Write run log to this path")
    return p


def _check_whisper_cache(model_name: str) -> None:
    model_name = str(model_name or "large-v3").strip()
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        weights = local_path / "model.bin"
        if weights.is_file() and weights.stat().st_size > 0:
            print(f"Whisper model cache: complete local model ({local_path})")
        else:
            print(f"[WARN] Whisper local model is incomplete: {weights} is missing or empty.")
        return
    repo_id = model_name if "/" in model_name else f"Systran/faster-whisper-{model_name}"
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_files_only=True)
        print(f"Whisper model cache: complete ({repo_id})")
    except Exception as exc:
        detail = " ".join(str(exc).split())[:320]
        print(
            f"[WARN] Whisper model cache is missing/incomplete for {repo_id}; "
            "the first transcription needs network access to finish downloading it. "
            f"Details: {type(exc).__name__}: {detail}"
        )


def _check(whisper_model: str = "large-v3") -> int:
    print("Dubbing runtime check")
    print(f"Python: {sys.version.split()[0]}")
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total, available = status.ullTotalPhys, status.ullAvailPhys
                print(f"System RAM: {total/1024**3:.2f} GiB total; {available/1024**3:.2f} GiB available; {(total-available)/1024**3:.2f} GiB used ({status.dwMemoryLoad}% load)")
        else:
            page = os.sysconf("SC_PAGE_SIZE"); total = os.sysconf("SC_PHYS_PAGES") * page
            available = os.sysconf("SC_AVPHYS_PAGES") * page
            print(f"System RAM: {total/1024**3:.2f} GiB total; {available/1024**3:.2f} GiB available; {(total-available)/1024**3:.2f} GiB used")
    except Exception as exc:
        print(f"System RAM: unavailable ({exc})")
    missing = []
    for binary in ("ffmpeg", "ffprobe"):
        found = shutil.which(binary)
        print(f"{binary}: {found or 'NOT FOUND'}")
        if not found:
            missing.append(binary)
    for package in ("whisperx", "demucs", "transformers", "voxcpm"):
        available = importlib.util.find_spec(package) is not None
        print(f"{package}: {'available' if available else 'NOT FOUND'}")
        if not available:
            missing.append(package)
    _check_whisper_cache(whisper_model)
    try:
        import torch
        available = torch.cuda.is_available()
        print(f"CUDA available: {available}")
        if available:
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {total:.2f} GiB total; {allocated:.2f} GiB allocated; {reserved:.2f} GiB reserved")
        else:
            print("GPU: unavailable")
    except Exception as exc:
        print(f"CUDA check failed: {type(exc).__name__}: {exc}")
        return 1
    return 1 if missing else 0


def _pid_is_running(pid) -> bool:
    """Conservatively check whether a lock owner's process still exists."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return True  # malformed ownership data: fail closed
    if pid <= 0:
        return True
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                # ERROR_INVALID_PARAMETER / ERROR_NOT_FOUND mean no such PID.
                return ctypes.get_last_error() not in (87, 1168)
            try:
                kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                kernel32.WaitForSingleObject.restype = ctypes.c_uint32
                kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel32.CloseHandle.restype = ctypes.c_int
                return kernel32.WaitForSingleObject(handle, 0) != 0  # WAIT_OBJECT_0 means exited
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True  # inability to inspect a process must never authorize deletion
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def _read_lock_payload(lock_path: Path):
    try:
        value = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _lock_owner_is_running(owner: dict) -> bool:
    """Treat either the CLI owner or any recorded worker PID as a live lock."""
    if not isinstance(owner, dict):
        return True
    children = owner.get("child_pids", [])
    if not isinstance(children, list):
        return True
    pids = [owner.get("pid"), *children]
    return any(_pid_is_running(pid) for pid in pids)


def _remove_stale_lock(lock_path: Path, expected_owner: dict) -> bool:
    """Atomically quarantine/delete a lock only if its unchanged owner PID is gone."""
    lock_path = Path(lock_path)
    current = _read_lock_payload(lock_path)
    if current is None or current != expected_owner or _lock_owner_is_running(current):
        return False
    # Re-read immediately before the atomic move to avoid deleting a replacement lock.
    if _read_lock_payload(lock_path) != current or _lock_owner_is_running(current):
        return False
    stale_path = lock_path.with_name(f"{lock_path.name}.stale.{os.getpid()}.{time.time_ns()}")
    try:
        os.replace(lock_path, stale_path)
    except OSError:
        return False
    moved = _read_lock_payload(stale_path)
    if moved != current or _lock_owner_is_running(current):
        try:
            if not lock_path.exists():
                os.replace(stale_path, lock_path)
        except OSError:
            pass
        return False
    try:
        stale_path.unlink()
    except OSError:
        return False
    return True


def _same_lock_workspace(owner: dict, workspace: Path) -> bool:
    value = owner.get("workspace")
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return os.path.normcase(os.path.abspath(value)) == os.path.normcase(os.path.abspath(str(workspace)))
    except (OSError, ValueError):
        return False


def _recover_stale_global_lock(global_lock: Path, owner: dict) -> bool:
    """Recover a dead global owner only when any matching local lock is also stale."""
    if not isinstance(owner, dict) or _lock_owner_is_running(owner):
        return False
    workspace_value = owner.get("workspace")
    if not isinstance(workspace_value, str) or not workspace_value.strip():
        return False
    workspace = Path(workspace_value)
    if not workspace.is_absolute():
        return False
    local_lock = workspace / "status" / "run.lock"
    if local_lock.exists():
        local_owner = _read_lock_payload(local_lock)
        if local_owner is None or not _same_lock_workspace(local_owner, workspace):
            return False
        if _lock_owner_is_running(local_owner):
            return False
        if not _remove_stale_lock(local_lock, local_owner):
            return False
    return _remove_stale_lock(global_lock, owner)


def _recover_stale_workspace_lock(lock: Path, workspace: Path) -> bool:
    owner = _read_lock_payload(lock)
    if owner is None or not _same_lock_workspace(owner, workspace):
        return False
    return _remove_stale_lock(lock, owner)


def _unlink_owned_lock(lock_path: Path, owner: dict) -> None:
    """Release only a lock that still contains this process's exact ownership record."""
    if _read_lock_payload(lock_path) != owner:
        return
    try:
        if _read_lock_payload(lock_path) == owner:
            Path(lock_path).unlink()
    except OSError:
        pass


def _runtime_lock(workspace: Path):
    workspace = Path(workspace)
    lock = workspace / "status" / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        global_dir = Path(os.environ["LOCALAPPDATA"]) / "DubbingControlCenter"
    else:
        global_dir = Path.home() / ".cache" / "dubbing-control-center"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_lock = global_dir / "global_run.lock"
    payload = {"pid": os.getpid(), "child_pids": [], "workspace": str(workspace), "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}

    try:
        global_fd = os.open(str(global_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read_lock_payload(global_lock) or {}
        if existing and _recover_stale_global_lock(global_lock, existing):
            print(f"[LOCK RECOVERY] Removed stale global lock for stopped PID {existing.get('pid')}.")
            try:
                global_fd = os.open(str(global_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = _read_lock_payload(global_lock) or {}
                raise SystemExit(f"Another dubbing job owns the global lock: PID {existing.get('pid', 'unknown')}, workspace {existing.get('workspace', 'unknown')}, started {existing.get('started_at', 'unknown')}. Wait for it or cancel this run. Do not remove the lock until the recorded process is verified stopped: {global_lock}")
        else:
            raise SystemExit(f"Another dubbing job owns the global lock: PID {existing.get('pid', 'unknown')}, workspace {existing.get('workspace', 'unknown')}, started {existing.get('started_at', 'unknown')}. Wait for it or cancel this run. Do not remove the lock until the recorded process is verified stopped: {global_lock}")
    with os.fdopen(global_fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read_lock_payload(lock) or {}
        if existing and _recover_stale_workspace_lock(lock, workspace):
            print(f"[LOCK RECOVERY] Removed stale workspace lock for stopped PID {existing.get('pid')}.")
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                existing = _read_lock_payload(lock) or {}
                _unlink_owned_lock(global_lock, payload)
                raise SystemExit(f"Workspace is locked by PID {existing.get('pid', 'unknown')} since {existing.get('started_at', 'unknown')}. Wait or cancel; verify the process before any manual unlock: {lock}")
        else:
            _unlink_owned_lock(global_lock, payload)
            raise SystemExit(f"Workspace is locked by PID {existing.get('pid', 'unknown')} since {existing.get('started_at', 'unknown')}. Wait or cancel; verify the process before any manual unlock: {lock}")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return lock, global_lock

def _set_active_child(locks, child_pid: int | None, *, stop_unconfirmed: bool = False) -> None:
    """Atomically record the currently running stage worker in both lock files."""
    snapshots = []
    for lock_path in locks:
        owner = _read_lock_payload(lock_path)
        if not owner or owner.get("pid") != os.getpid():
            raise RuntimeError(f"Cannot update a lock not owned by this process: {lock_path}")
        snapshots.append((Path(lock_path), owner))
    if not snapshots:
        return
    identity = {(owner.get("pid"), owner.get("workspace"), owner.get("started_at")) for _, owner in snapshots}
    if len(identity) != 1:
        raise RuntimeError("Global and workspace lock ownership data do not match.")
    for lock_path, owner in snapshots:
        updated = dict(owner)
        updated["child_pids"] = [int(child_pid)] if child_pid is not None else []
        updated["stop_unconfirmed"] = bool(stop_unconfirmed)
        fd, temporary_name = tempfile.mkstemp(prefix=lock_path.name + ".", suffix=".tmp", dir=str(lock_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(updated, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, lock_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _release_owned_locks(locks) -> None:
    """Release this CLI's locks only after its recorded worker tree is gone."""
    for lock_path in locks:
        owner = _read_lock_payload(lock_path)
        if not owner or owner.get("pid") != os.getpid():
            continue
        children = owner.get("child_pids", [])
        if not isinstance(children, list) or any(_pid_is_running(pid) for pid in children):
            print(f"[LOCK] Retaining {lock_path}: worker shutdown was not confirmed.")
            continue
        _unlink_owned_lock(lock_path, owner)


def _run(args: argparse.Namespace) -> int:
    if args.profile:
        profile = _profiles().get(args.profile)
        if not isinstance(profile, dict):
            raise SystemExit(f"Profile not found: {args.profile}")
        explicit = getattr(args, "explicit_options", set())
        for key, value in profile.items():
            option = "--" + key.replace("_", "-")
            if hasattr(args, key) and key not in {"input", "open_tui", "profile", "save_profile", "batch_list", "batch_dir", "recursive"} and option not in explicit:
                setattr(args, key, value)

    batch_requested = bool(args.batch_list or args.batch_dir)
    if batch_requested:
        if args.input:
            raise SystemExit("Use either one input file or a batch source, not both.")
        if args.recursive and not args.batch_dir:
            raise SystemExit("--recursive is only valid with --batch-dir.")
        sources = _batch_sources(args)
        if args.save_profile:
            values = {k: v for k, v in vars(args).items() if k not in {"input", "open_tui", "profile", "save_profile", "list_profiles", "check", "dry_run", "resume", "redub", "log_file", "output_dir", "explicit_options", "batch_list", "batch_dir", "recursive"}}
            _save_profile(args.save_profile, values)
            print(f"Profile saved: {args.save_profile}")
        batch_order = ("each video completes Transcribe -> Translate -> Dub before the next" if args.task == "full" else f"complete {args.task} for each video before the next")
        print(f"BATCH: {len(sources)} local media item(s), serial mode; {batch_order}.")
        succeeded: list[str] = []
        failed: list[tuple[str, str]] = []
        stopped = False
        for index, source in enumerate(sources, 1):
            print(f"\n[BATCH START {index}/{len(sources)}] {source}")
            child = argparse.Namespace(**vars(args))
            child.input = source
            child.batch_list = None
            child.batch_dir = None
            child.recursive = False
            child.profile = None
            child.save_profile = None
            child.log_file = None
            if args.output_dir:
                child.output_dir = str(_batch_destination(args.output_dir, source))
            try:
                code = _run(child)
                reason = f"exit code {code}"
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                reason = str(exc) if not isinstance(exc.code, int) else f"exit code {code}"
                if reason:
                    print(f"[BATCH ITEM ERROR] {reason}")
            except Exception as exc:
                code = 1
                reason = f"{type(exc).__name__}: {exc}"
                print(f"[BATCH ITEM ERROR] {reason}")
            if code == 0:
                succeeded.append(source)
                print(f"[BATCH ITEM DONE] {source}")
            else:
                failed.append((source, "stopped by user; checkpoint kept" if code == 130 else reason))
                print(f"[BATCH ITEM FAILED] {source} ({reason})")
                if code == 130:
                    stopped = True
                    break
        print(f"\nBATCH SUMMARY: {len(succeeded)} succeeded; {len(failed)} failed; {len(sources)} total.")
        for source, reason in failed:
            print(f"  FAILED: {source} — {reason}")
        return 130 if stopped else (1 if failed else 0)
    if args.recursive:
        raise SystemExit("--recursive is only valid with --batch-dir.")

    raw_input = args.input
    if not raw_input and args.save_profile:
        values = {k: v for k, v in vars(args).items() if k not in {"input", "open_tui", "profile", "save_profile", "list_profiles", "check", "dry_run", "resume", "redub", "log_file", "output_dir", "explicit_options"}}
        _save_profile(args.save_profile, values)
        print(f"Profile saved: {args.save_profile}")
        return 0
    if not raw_input:
        raise SystemExit("A local input file is required.")
    if _is_url(raw_input):
        raise SystemExit("ERROR: HTTP/HTTPS URLs are not supported. Provide a local file path.")
    source = Path(raw_input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"ERROR: Local input file not found: {source}")
    if args.reference_audio and (_is_url(args.reference_audio) or not Path(args.reference_audio).expanduser().is_file()):
        raise SystemExit("ERROR: Reference audio must be an existing local file.")
    if args.external_srt and (_is_url(args.external_srt) or not Path(args.external_srt).expanduser().is_file()):
        raise SystemExit("ERROR: External SRT must be an existing local file.")
    if args.task == "srt_dub" and not args.external_srt:
        raise SystemExit("ERROR: --task srt_dub requires --external-srt.")
    if args.chunk_minutes <= 0 or args.split_threshold_seconds < 0 or args.search_window < 0:
        raise SystemExit("ERROR: Split settings must be positive (threshold may be zero).")
    if args.redub and args.task not in {"full", "dub", "srt_dub"}:
        raise SystemExit("ERROR: --redub can only be used with --task full or --task dub.")
    if not 1.0 <= args.max_speed <= 2.5:
        raise SystemExit("ERROR: --max-speed must be between 1.0 and 2.5.")

    values = {k: v for k, v in vars(args).items() if k not in {"input", "open_tui", "profile", "save_profile", "list_profiles", "check", "dry_run", "resume", "redub", "log_file", "output_dir", "explicit_options"}}
    if args.save_profile:
        _save_profile(args.save_profile, values)
        print(f"Profile saved: {args.save_profile}")

    workspace = _workspace(source)
    final = workspace / "final"
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else workspace / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    base_cmd = [sys.executable, str(BASE / "dub.py"), str(source), "--source-language", args.source_language,
           "--target-language", args.target_language, "--target-code", args.target_code, "--llm", args.llm,
           "--whisper", args.whisper, "--device", args.device, "--gender", args.gender, "--output", args.output,
           "--whisper-batch-size", str(args.whisper_batch_size),
           "--split-mode", args.split_mode, "--chunk-minutes", str(args.chunk_minutes),
           "--split-threshold-seconds", str(args.split_threshold_seconds), "--search-window", str(args.search_window),
           "--max-speed", str(args.max_speed), "--text-mode", args.text_mode]
    if args.reference_audio:
        base_cmd += ["--reference-audio", str(Path(args.reference_audio).expanduser().resolve())]
    if args.external_srt:
        base_cmd += ["--external-srt", str(Path(args.external_srt).expanduser().resolve())]
    tasks = ["transcribe", "translate", "dub"] if args.task == "full" else [args.task]
    commands = []
    for task in tasks:
        cmd = base_cmd + ["--task", task]
        if args.redub and task == "dub":
            cmd.append("--redub")
        commands.append(cmd)

    print(f"Workspace: {workspace}")
    print(f"Final output directory: {final}")
    print("Operation sequence:")
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. " + subprocess.list2cmdline(cmd))
    if args.dry_run:
        print("DRY RUN: no models were loaded and no processing was started.")
        return 0

    workspace.mkdir(parents=True, exist_ok=True)
    locks = _runtime_lock(workspace)
    os.environ["ZAST_WORK_DIR"] = str(workspace)
    os.environ["ZAST_TEMP_DIR"] = str(workspace / "temp")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    worker_state = {"process": None}
    worker_guard = threading.RLock()
    def on_console_close(control_type):
        if int(control_type) not in (2, 5, 6):  # close, logoff, shutdown
            return False
        with worker_guard:
            active = worker_state["process"]
            if active is not None and active.poll() is None:
                stopped = terminate_process_tree(active, timeout=2.0)
                try:
                    if stopped:
                        _set_active_child(locks, None)
                        worker_state["process"] = None
                    else:
                        _set_active_child(locks, active.pid, stop_unconfirmed=True)
                except Exception:
                    pass
        return True
    console_handler = install_console_close_handler(on_console_close)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            for cmd in commands:
                log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n")
            log.write(f"WORKSPACE: {workspace}\n")
            log.flush()
            code = 0
            for phase, cmd in zip(tasks, commands):
                print(f"\n[PROCESS START] {phase.title()} (separate Python process)")
                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
                proc = subprocess.Popen(cmd, cwd=BASE, env=child_env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                assert proc.stdout is not None
                try:
                    with worker_guard:
                        worker_state["process"] = proc
                        _set_active_child(locks, proc.pid)
                    last_log_flush = time.monotonic()
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        now = time.monotonic()
                        if now - last_log_flush >= 1.0:
                            log.flush()
                            last_log_flush = now
                    code = proc.wait()
                    with worker_guard:
                        _set_active_child(locks, None)
                        worker_state["process"] = None
                except KeyboardInterrupt:
                    log.write("\n[INTERRUPTED] Ctrl+C received; stopping the active worker tree and preserving completed checkpoints.\n")
                    log.flush()
                    print("\n[STOP] Ctrl+C received. Stopping the active worker and its FFmpeg descendants; completed chunks are kept.")
                    with worker_guard:
                        stopped = terminate_process_tree(proc)
                        if stopped:
                            _set_active_child(locks, None)
                            worker_state["process"] = None
                            print("[STOP] Worker tree stopped; checkpoints are safe to resume.")
                        else:
                            _set_active_child(locks, proc.pid, stop_unconfirmed=True)
                            print("[STOP WARNING] Worker-tree shutdown could not be confirmed. The lock is retained to prevent a duplicate GPU job.")
                    code = 130
                log.flush()
                print(f"[PROCESS EXIT] {phase.title()} code={code}")
                if code:
                    break
        elapsed = str(timedelta(seconds=int(time.perf_counter() - started)))
        print(f"Elapsed: {elapsed}\nFull log: {log_path}")
        if code:
            return code
        if args.output_dir:
            dest = Path(args.output_dir).expanduser().resolve()
            dest.mkdir(parents=True, exist_ok=True)
            for item in final.iterdir() if final.exists() else []:
                if item.is_file():
                    shutil.copy2(item, dest / item.name)
        return 0
    finally:
        remove_console_close_handler(console_handler)
        _release_owned_locks(locks)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except (OSError, ValueError):
                pass
    p = _parser()
    args = p.parse_args()
    args.explicit_options = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    if args.open_tui:
        return subprocess.call([sys.executable, str(BASE / "dubbing_tui.py")], cwd=BASE)
    if args.check:
        return _check(args.whisper)
    if args.list_languages:
        print("\n".join(LANGUAGES)); return 0
    if args.list_voices:
        print("\n".join(VOICES)); return 0
    if args.list_tts_backends:
        catalog = get_backend_catalog(args.target_code)
        if catalog is None:
            for name in get_backend_names():
                print(f"{name}: availability metadata not provided by this factory version")
        else:
            for item in catalog:
                print(f"{item['name']}: {item['status']} (languages={item['languages']})")
        return 0
    if args.list_profiles:
        print("\n".join(_profiles()) or "No saved profiles."); return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
