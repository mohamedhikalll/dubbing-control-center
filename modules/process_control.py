from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def _windows_process_tree(root_pid: int) -> list[int] | None:
    """Return root and descendants, or None if Windows process enumeration failed."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESSENTRY32W)]
        kernel32.Process32FirstW.restype = ctypes.c_int
        kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PROCESSENTRY32W)]
        kernel32.Process32NextW.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            return None
        try:
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            parent_by_pid: dict[int, int] = {}
            if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                while True:
                    parent_by_pid[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                    entry.dwSize = ctypes.sizeof(entry)
                    if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                        break
            descendants: set[int] = set()
            frontier = {int(root_pid)}
            while frontier:
                found = {pid for pid, parent in parent_by_pid.items() if parent in frontier and pid != root_pid}
                found -= descendants
                if not found:
                    break
                descendants.update(found)
                frontier = found
            # Stop parent first to prevent additional children from being spawned.
            return [int(root_pid), *sorted(descendants, reverse=True)]
        finally:
            kernel32.CloseHandle(snapshot)
    except Exception:
        return None


def _windows_pid_running(pid: int) -> bool:
    """Conservative liveness check for a process ID using a waitable handle."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
        if not handle:
            return ctypes.get_last_error() not in (87, 1168)
        try:
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            return kernel32.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.CloseHandle(handle)
    except Exception:
        return True


def _terminate_windows_tree(process, timeout: float) -> bool:
    pids = _windows_process_tree(process.pid)
    if pids is None:
        return False
    failures = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
        except ProcessLookupError:
            pass
        except OSError:
            if _windows_pid_running(pid):
                failures.append(pid)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        still_running = [pid for pid in pids if _windows_pid_running(pid)]
        if not still_running:
            return not failures
        time.sleep(0.1)
    return False



def install_console_close_handler(callback):
    """Register a retained callback for Windows console close/logoff/shutdown events."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32)
        native_callback = handler_type(callback)
        register = kernel32.SetConsoleCtrlHandler
        register.argtypes = [handler_type, ctypes.c_int]
        register.restype = ctypes.c_int
        if not register(native_callback, True):
            return None
        return kernel32, native_callback, register
    except Exception:
        return None


def remove_console_close_handler(registration) -> None:
    """Unregister a previously installed console handler, if any."""
    if not registration:
        return
    kernel32, native_callback, register = registration
    try:
        register(native_callback, False)
    except Exception:
        pass


def terminate_process_tree(process, *, timeout: float = 12.0) -> bool:
    """Stop a worker and its descendants; return True only when shutdown is confirmed.

    Windows process enumeration and termination are performed directly through the
    OS APIs, avoiding a dependency on taskkill permissions. If shutdown cannot be
    confirmed, callers must retain the run lock rather than start a duplicate job.
    """
    if process is None:
        return True
    try:
        if process.poll() is not None:
            return True
    except Exception:
        return False

    if os.name == "nt":
        return _terminate_windows_tree(process, timeout)

    try:
        process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return False
    except OSError:
        return False
    return process.poll() is not None
