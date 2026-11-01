from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
import math
import os
import struct
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dub_cli import _batch_destination, _check_whisper_cache, _runtime_lock, _lock_owner_is_running, _set_active_child, _release_owned_locks
import dubbing_tui
from dubbing_tui import classify_log_line
from modules.process_control import terminate_process_tree, install_console_close_handler, remove_console_close_handler
from long_video_pipeline.splitter import calculate_boundaries, detect_silences, is_valid_audio_chunk, split_video
from long_video_pipeline.job_manager import JobContext


def _load_bilingual_writer():
    """Load the pure SRT writer without importing model-heavy dub.py modules."""
    source = (ROOT / "dub.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_bilingual_srt"
    )
    namespace = {
        "json": json,
        "Path": Path,
        "load_manifest": lambda job: job.manifest,
        "timestamp": lambda seconds: _timestamp(seconds),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "dub.py", "exec"), namespace)
    return namespace["write_bilingual_srt"]


def _timestamp(seconds: float) -> str:
    total = int(round(seconds * 1000))
    h, rest = divmod(total, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class BoundaryTests(unittest.TestCase):
    def test_only_searches_local_windows_and_uses_quiet_gap(self):
        calls = []

        def search(start, end):
            calls.append((start, end))
            return [
                (a, b) for a, b in ((59.0, 61.0), (119.0, 121.0))
                if b >= start and a <= end
            ]

        boundaries = calculate_boundaries(
            140.0, [], chunk_minutes=1.0, search_window=10.0,
            silence_search=search,
        )
        self.assertEqual(calls, [(50.0, 70.0), (110.0, 130.0)])
        self.assertEqual(boundaries, [0.0, 60.0, 120.0, 140.0])

    def test_audio_chunk_validator_rejects_corrupt_nonempty_file(self):
        with tempfile.TemporaryDirectory(prefix="dub_corrupt_audio_test_") as temp:
            path = Path(temp) / "chunk.m4a"
            path.write_bytes(b"nonempty but not a media file")
            with patch("long_video_pipeline.splitter._run", return_value=subprocess.CompletedProcess([], 1, "", "invalid")):
                self.assertFalse(is_valid_audio_chunk(path, 10.0))

    def test_split_rebuilds_nonempty_corrupt_chunk_and_promotes_atomically(self):
        with tempfile.TemporaryDirectory(prefix="dub_split_resume_test_") as temp:
            root = Path(temp)
            source = root / "source.mp4"
            source.write_bytes(b"test source placeholder")
            job = JobContext(root, job_id="source_dubbing")
            damaged = job.chunks_dir / "chunk_001.m4a"
            damaged.write_bytes(b"partial ffmpeg output")

            def fake_write(_source, destination, _start, _end):
                Path(destination).write_bytes(b"VALID-CHUNK")

            def valid_chunk(candidate, _duration, **_kwargs):
                path = Path(candidate)
                return path.is_file() and path.read_bytes() == b"VALID-CHUNK"

            with patch("long_video_pipeline.splitter._require_tool"),                  patch("long_video_pipeline.splitter.probe_duration", return_value=90.0),                  patch("long_video_pipeline.splitter.detect_silences", return_value=[]),                  patch("long_video_pipeline.splitter._write_chunk", side_effect=fake_write),                  patch("long_video_pipeline.splitter.is_valid_audio_chunk", side_effect=valid_chunk):
                chunks = split_video(source, job, chunk_minutes=1.0, search_window=0.0)

            self.assertEqual(len(chunks), 2)
            self.assertEqual(damaged.read_bytes(), b"VALID-CHUNK")
            self.assertTrue(all(Path(chunk.path).read_bytes() == b"VALID-CHUNK" for chunk in chunks))
            self.assertFalse(any("partial" in item.name for item in job.chunks_dir.iterdir()))
            self.assertEqual(json.loads((job.status_dir / "chunks_manifest.json").read_text(encoding="utf-8"))["chunk_minutes"], 1.0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_ffmpeg_detects_absolute_times_after_local_seek(self):
        with tempfile.TemporaryDirectory(prefix="dub_silence_test_") as temp:
            wav_path = Path(temp) / "source.wav"
            sample_rate = 8000
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                frames = bytearray()
                for i in range(70 * sample_rate):
                    second = i / sample_rate
                    silent = 29.0 <= second <= 31.0
                    sample = 0 if silent else int(5000 * math.sin(2 * math.pi * 440 * second))
                    frames.extend(struct.pack("<h", sample))
                handle.writeframes(frames)
            found = detect_silences(
                wav_path, minimum_silence=0.25, silence_threshold_db=-30,
                search_start=20.0, search_end=40.0,
            )
            self.assertTrue(any(28.5 <= start <= 30.0 and 30.0 <= end <= 31.5 for start, end in found), found)


class RuntimeLockTests(unittest.TestCase):
    def test_reclaims_dead_global_and_workspace_lock(self):
        with tempfile.TemporaryDirectory(prefix="dub_stale_lock_test_") as temp:
            root = Path(temp)
            workspace = root / "workspace"
            status = workspace / "status"
            status.mkdir(parents=True)
            local_app = root / "appdata"
            global_dir = local_app / "DubbingControlCenter"
            global_dir.mkdir(parents=True)
            owner = {"pid": 987654321, "workspace": str(workspace), "started_at": "2026-01-01T00:00:00+00:00"}
            local_lock = status / "run.lock"
            global_lock = global_dir / "global_run.lock"
            local_lock.write_text(json.dumps(owner), encoding="utf-8")
            global_lock.write_text(json.dumps(owner), encoding="utf-8")
            with patch.dict(os.environ, {"LOCALAPPDATA": str(local_app)}), patch("dub_cli._pid_is_running", return_value=False):
                acquired_local, acquired_global = _runtime_lock(workspace)
            try:
                self.assertTrue(acquired_local.is_file())
                self.assertTrue(acquired_global.is_file())
                self.assertEqual(json.loads(acquired_local.read_text(encoding="utf-8"))["pid"], os.getpid())
                self.assertEqual(json.loads(acquired_global.read_text(encoding="utf-8"))["pid"], os.getpid())
            finally:
                acquired_local.unlink(missing_ok=True)
                acquired_global.unlink(missing_ok=True)

    def test_active_worker_pid_is_recorded_and_prevents_early_unlock(self):
        with tempfile.TemporaryDirectory(prefix="dub_worker_lock_test_") as temp:
            root = Path(temp)
            workspace = root / "workspace"
            local_app = root / "appdata"
            with patch.dict(os.environ, {"LOCALAPPDATA": str(local_app)}):
                locks = _runtime_lock(workspace)
                _set_active_child(locks, 987654321)
                for lock in locks:
                    self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["child_pids"], [987654321])
                with patch("dub_cli._pid_is_running", side_effect=lambda pid: int(pid) == 987654321):
                    _release_owned_locks(locks)
                    self.assertTrue(all(lock.exists() for lock in locks))
                with patch("dub_cli._pid_is_running", return_value=False):
                    _release_owned_locks(locks)
                    self.assertTrue(all(not lock.exists() for lock in locks))

    def test_does_not_reclaim_lock_when_owner_may_be_alive(self):
        with tempfile.TemporaryDirectory(prefix="dub_live_lock_test_") as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            local_app = root / "appdata"
            global_dir = local_app / "DubbingControlCenter"
            global_dir.mkdir(parents=True)
            owner = {"pid": 987654321, "workspace": str(workspace), "started_at": "2026-01-01T00:00:00+00:00"}
            global_lock = global_dir / "global_run.lock"
            global_lock.write_text(json.dumps(owner), encoding="utf-8")
            with patch.dict(os.environ, {"LOCALAPPDATA": str(local_app)}), patch("dub_cli._pid_is_running", return_value=True):
                with self.assertRaisesRegex(SystemExit, "Another dubbing job owns the global lock"):
                    _runtime_lock(workspace)
            self.assertTrue(global_lock.is_file())
            self.assertFalse((workspace / "status" / "run.lock").exists())


class LastSessionTests(unittest.TestCase):
    def test_tui_restores_input_mode_and_settings(self):
        with tempfile.TemporaryDirectory(prefix="dub_last_session_test_") as temp:
            path = Path(temp) / "last_session.json"
            path.write_text(json.dumps({"input": "D:/lectures", "source_mode": "folder", "task": "full"}), encoding="utf-8")
            with patch.object(dubbing_tui, "LAST_SESSION_FILE", path):
                settings = dubbing_tui.defaults()
            self.assertEqual(settings["input"], "D:/lectures")
            self.assertEqual(settings["source_mode"], "folder")
            self.assertEqual(settings["task"], "full")


class ResumeWorkerTests(unittest.TestCase):
    def test_live_worker_pid_keeps_owner_lock_active(self):
        owner = {"pid": 111, "child_pids": [222]}
        with patch("dub_cli._pid_is_running", side_effect=lambda pid: int(pid) == 222):
            self.assertTrue(_lock_owner_is_running(owner))

    def test_dead_cli_and_worker_can_be_recovered(self):
        owner = {"pid": 111, "child_pids": [222]}
        with patch("dub_cli._pid_is_running", return_value=False):
            self.assertFalse(_lock_owner_is_running(owner))

    @unittest.skipUnless(os.name == "nt", "Windows process-tree APIs")
    def test_termination_stops_the_recorded_windows_process_tree(self):
        class FakeProcess:
            pid = 43210
            code = None
            def poll(self): return self.code
            def wait(self, timeout=None): self.code = 0; return 0
        proc = FakeProcess()
        with patch("modules.process_control._windows_process_tree", return_value=[43210, 43212]), \
             patch("modules.process_control.os.kill") as kill, \
             patch("modules.process_control._windows_pid_running", return_value=False):
            self.assertTrue(terminate_process_tree(proc))
        self.assertEqual([call.args[0] for call in kill.call_args_list], [43210, 43212])

    @unittest.skipUnless(os.name == "nt", "Windows process-tree APIs")
    def test_unenumerable_windows_tree_is_not_reported_as_safe(self):
        class FakeProcess:
            pid = 43211
            code = None
            def poll(self): return self.code
            def wait(self, timeout=None): self.code = 0; return 0
        with patch("modules.process_control._windows_process_tree", return_value=None):
            self.assertFalse(terminate_process_tree(FakeProcess()))


class ConsoleCloseHandlerTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows console close events")
    def test_console_close_handler_registers_invokes_and_unregisters(self):
        events = []
        registration = install_console_close_handler(lambda event: events.append(int(event)) or True)
        self.assertIsNotNone(registration)
        try:
            self.assertEqual(registration[1](2), 1)
            self.assertEqual(events, [2])
        finally:
            remove_console_close_handler(registration)


class TUILogTests(unittest.TestCase):
    def test_torchcodec_warning_block_is_not_an_error(self):
        warning_state = False
        collecting_failure = False
        lines = [
            "[start of libtorchcodec loading traceback]",
            "FFmpeg version 7: Could not find module libtorchcodec_core7.dll",
            "[end of libtorchcodec loading traceback].",
            "warnings.warn(message)",
        ]
        for line in lines:
            result = classify_log_line(line, warning_state, collecting_failure)
            self.assertFalse(result["error"], (line, result))
            warning_state = result["warning_traceback"]
            collecting_failure = result["collecting_failure"]

    def test_real_stage_failure_is_reported(self):
        result = classify_log_line("[STAGE FAILED] transcribe: model missing", False, False)
        self.assertTrue(result["error"])
        self.assertTrue(result["collecting_failure"])
        self.assertTrue(classify_log_line("RuntimeError: missing model", False, True)["error"])


class CLIAndBatchTests(unittest.TestCase):
    def test_whisper_cache_check_reports_incomplete_snapshot(self):
        fake_hub = types.ModuleType("huggingface_hub")

        def fail_local_lookup(**kwargs):
            self.assertTrue(kwargs["local_files_only"])
            raise OSError("model.bin is missing")

        fake_hub.snapshot_download = fail_local_lookup
        output = io.StringIO()
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with contextlib.redirect_stdout(output):
                _check_whisper_cache("large-v3")
        self.assertIn("missing/incomplete", output.getvalue())
        self.assertIn("network access", output.getvalue())

    def test_batch_dry_run_finishes_video_one_before_starting_video_two(self):
        with tempfile.TemporaryDirectory(prefix="dub_serial_batch_test_") as temp:
            folder = Path(temp)
            first = folder / "01_first.mp4"
            second = folder / "02_second.mp4"
            first.write_bytes(b"")
            second.write_bytes(b"")
            result = subprocess.run(
                [sys.executable, str(ROOT / "dub_cli.py"), "--batch-dir", str(folder), "--dry-run"],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            start_one = result.stdout.index("[BATCH START 1/2]")
            start_two = result.stdout.index("[BATCH START 2/2]")
            done_positions = [i for i in range(len(result.stdout)) if result.stdout.startswith("[BATCH ITEM DONE]", i)]
            self.assertEqual(len(done_positions), 2, result.stdout)
            self.assertLess(start_one, done_positions[0], result.stdout)
            self.assertLess(done_positions[0], start_two, result.stdout)
            self.assertLess(start_two, done_positions[1], result.stdout)
            self.assertIn(first.name, result.stdout[start_one:start_two])
            self.assertIn(second.name, result.stdout[start_two:])
            self.assertIn("Transcribe -> Translate -> Dub before the next", result.stdout)

    def test_batch_exports_do_not_collide_on_same_stem(self):
        with tempfile.TemporaryDirectory(prefix="dub_batch_test_") as temp:
            root = Path(temp)
            one = root / "folder_a" / "lecture.mp4"
            two = root / "folder_b" / "lecture.mp4"
            first = _batch_destination(root / "exports", one)
            second = _batch_destination(root / "exports", two)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("lecture_"))

    def test_cli_help_does_not_eagerly_import_tts_factory(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "dub_cli.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Local-file dubbing", completed.stdout)
        self.assertNotIn("modules.tts_backends.factory", completed.stderr)


class BilingualSRTTests(unittest.TestCase):
    def _job(self, root: Path, translated_segments):
        transcription = root / "transcription"
        translation = root / "translation"
        transcription.mkdir(parents=True)
        translation.mkdir(parents=True)
        source_segments = [{"start": 2.0, "end": 3.2, "text": "The original English sentence."}]
        (transcription / "chunk_001.json").write_text(
            json.dumps({"segments": source_segments}, ensure_ascii=False), encoding="utf-8"
        )
        (translation / "chunk_001.json").write_text(
            json.dumps({"segments": translated_segments}, ensure_ascii=False), encoding="utf-8"
        )
        return SimpleNamespace(
            transcription_dir=transcription,
            translation_dir=translation,
            manifest=[SimpleNamespace(number=1, start=60.0, end=120.0)],
        )

    def test_bilingual_srt_keeps_source_text_and_timing(self):
        writer = _load_bilingual_writer()
        with tempfile.TemporaryDirectory(prefix="dub_srt_test_") as temp:
            root = Path(temp)
            job = self._job(root, [{"start": 2.0, "end": 3.2, "normal_text": "الجملة العربية الأصلية."}])
            output = root / "final" / "bilingual.srt"
            self.assertEqual(writer(job, output), 1)
            content = output.read_text(encoding="utf-8")
            self.assertIn("00:01:02,000 --> 00:01:03,200", content)
            self.assertIn("EN: The original English sentence.", content)
            self.assertIn("AR: الجملة العربية الأصلية.", content)

    def test_bilingual_srt_refuses_different_cue_counts(self):
        writer = _load_bilingual_writer()
        with tempfile.TemporaryDirectory(prefix="dub_srt_mismatch_") as temp:
            root = Path(temp)
            job = self._job(root, [])
            with self.assertRaisesRegex(RuntimeError, "alignment failed"):
                writer(job, root / "bilingual.srt")


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "numpy", "soundfile", "librosa")),
    "Optional runtime audio dependencies are not installed",
)
class AudioCacheTests(unittest.TestCase):
    def test_cache_key_changes_with_text_voice_gender_language_and_speed(self):
        from modules.time_sync import TimeSync

        class Engine:
            pass

        with tempfile.TemporaryDirectory(prefix="dub_cache_test_") as temp:
            root = Path(temp)
            voice_a = root / "a.wav"
            voice_b = root / "b.wav"
            voice_a.write_bytes(b"a")
            voice_b.write_bytes(b"b")
            syncer = TimeSync(Engine())
            base = syncer._audio_cache_tag("text", "Arabic", 1.0, 2.0, str(voice_a), "Man", 2.5)
            self.assertNotEqual(base, syncer._audio_cache_tag("changed", "Arabic", 1.0, 2.0, str(voice_a), "Man", 2.5))
            self.assertNotEqual(base, syncer._audio_cache_tag("text", "Arabic", 1.0, 2.0, str(voice_b), "Man", 2.5))
            self.assertNotEqual(base, syncer._audio_cache_tag("text", "Arabic", 1.0, 2.0, str(voice_a), "Woman", 2.5))
            self.assertNotEqual(base, syncer._audio_cache_tag("text", "English", 1.0, 2.0, str(voice_a), "Man", 2.5))
            self.assertNotEqual(base, syncer._audio_cache_tag("text", "Arabic", 1.0, 2.0, str(voice_a), "Man", 2.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
