from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import timedelta
from pathlib import Path

from modules.process_control import terminate_process_tree, install_console_close_handler, remove_console_close_handler

BASE = Path(__file__).resolve().parent
PROFILE_FILE = BASE / "profiles.json"
LAST_SESSION_FILE = BASE / "last_session.json"
CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; WHITE="\033[97m"; DIM="\033[90m"; RESET="\033[0m"
LANGS=[("Auto detect","auto"),("English","en"),("Arabic","ar"),("French","fr"),("Spanish","es"),("German","de"),("Italian","it"),("Portuguese","pt"),("Japanese","ja"),("Korean","ko"),("Chinese","zh"),("Russian","ru"),("Hindi","hi")]
TARGETS=[("Arabic","Arabic","ar"),("English","English","en"),("French","French","fr"),("Spanish","Spanish","es"),("German","German","de"),("Italian","Italian","it"),("Portuguese","Portuguese","pt"),("Japanese","Japanese","ja"),("Korean","Korean","ko"),("Chinese","Chinese","zh"),("Russian","Russian","ru"),("Hindi","Hindi","hi")]
LLMS=[("Fanar-1-9B-Instruct","Fanar-1-9B-Instruct (Arabic/English — 8 GB+ VRAM)"),("Qwen3.5-9B","Qwen3.5-9B"),("Qwen2.5-7B-Instruct","Qwen2.5-7B-Instruct")]
WHISPERS=["large-v3","medium","small","base","tiny"]
MEDIA_EXT={".mp4",".mkv",".mov",".avi",".wmv",".webm",".m4v",".mpeg",".mpg",".ts",".m2ts",".3gp",".flv",".mp3",".wav",".m4a",".flac",".ogg",".aac",".opus"}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def profiles():
    try:
        value=json.loads(PROFILE_FILE.read_text(encoding="utf-8")); return value if isinstance(value,dict) else {}
    except (OSError,json.JSONDecodeError): return {}


def save_profiles(data): PROFILE_FILE.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def defaults():
    settings = {"input":"","source_mode":"single","recursive":False,"source_language":"auto","target_language":"Arabic","target_code":"ar","llm":LLMS[0][1],"whisper":"large-v3","whisper_batch_size":4,"device":"cuda","gender":"Man","output":"video","reference_audio":"","external_srt":"","split_mode":"auto","chunk_minutes":10.0,"split_threshold_seconds":1140.0,"search_window":30.0,"max_speed":2.5,"task":"full","text_mode":"normal"}
    try:
        saved = json.loads(LAST_SESSION_FILE.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            settings.update({key: value for key, value in saved.items() if key in settings})
    except (OSError, json.JSONDecodeError):
        pass
    return settings


def header():
    print(CYAN + "=" * 76 + RESET)
    print(GREEN + "                         DUBBING CONTROL CENTER" + RESET)
    print(CYAN + "=" * 76 + RESET)


def pause(): input("\nPress Enter to continue...")


def input_inventory(s):
    """Return the files that the selected batch mode will ask the CLI to process."""
    source=Path(s["input"]).expanduser()
    if s["source_mode"]=="single":
        return [source] if source.is_file() else []
    if s["source_mode"]=="list":
        if not source.is_file(): return []
        files=[]
        for raw in source.read_text(encoding="utf-8-sig").splitlines():
            value=raw.strip().strip('"')
            if not value or value.startswith("#") or "://" in value: continue
            path=Path(value).expanduser()
            files.append(path if path.is_absolute() else source.parent/path)
        return files
    if not source.is_dir(): return []
    paths=source.rglob("*") if s["recursive"] else source.iterdir()
    return sorted([path for path in paths if path.is_file() and path.suffix.lower() in MEDIA_EXT], key=lambda p:str(p).casefold())


def classify_log_line(line, warning_traceback=False, collecting_failure=False):
    """Separate warning-library tracebacks from actual processing failures."""
    low = line.lower()
    starts_warning_traceback = "[start of libtorchcodec loading traceback]" in low
    in_warning_traceback = warning_traceback or starts_warning_traceback
    is_warning = not in_warning_traceback and ("warning" in low or "warn:" in low)
    ends_warning_traceback = "[end of libtorchcodec loading traceback]" in low
    failure_marker = (
        "[stage failed]" in low
        or re.search(
            r"(?:^|\s)(?:error|critical):|: error:|\b(?:runtimeerror|importerror|valueerror|typeerror|filenotfounderror|exception):",
            low,
        )
    )
    is_error = not in_warning_traceback and not is_warning and (failure_marker or collecting_failure)
    return {
        "warning": is_warning,
        "error": bool(is_error),
        "warning_traceback": False if ends_warning_traceback else in_warning_traceback,
        "collecting_failure": collecting_failure or "[stage failed]" in low,
    }


def _progress_bar(done, total, width=24):
    try:
        total = max(1, int(total))
        done = max(0, min(total, int(done)))
    except (TypeError, ValueError):
        done, total = 0, 1
    filled = int(width * done / total)
    return "#" * filled + "-" * (width - filled)


def _stage_name(value):
    text = str(value or "").lower()
    for token, label in (("split", "Split"), ("transcrib", "Transcribe"), ("translat", "Translate"),
                         ("separat", "Separate"), ("dub", "Dub"), ("assembl", "Assemble"), ("packag", "Package")):
        if token in text:
            return label
    return str(value or "Starting")


def render_dashboard(*, total_videos, completed_videos, failed_videos, current_video,
                     current_path, inventory, stage, stage_percent, chunk,
                     remaining_steps, activity, speed, elapsed, model_status,
                     model_percent, model_error, warnings, errors, last, recent,
                     batch_mode):
    """Render a compact dashboard while preserving the existing ANSI-only TUI."""
    clear(); header()
    remaining = max(0, total_videos - completed_videos - failed_videos)
    current_label = Path(str(current_path)).name or str(current_path)
    stage_label = _stage_name(stage)
    print(f"\n{WHITE}BATCH STATUS{RESET}  {GREEN}{completed_videos} done{RESET}  {RED}{failed_videos} failed{RESET}  {YELLOW}{remaining} remaining{RESET} / {total_videos} total")
    print(f"{DIM}{'-' * 76}{RESET}")
    print(f"{WHITE}CURRENT VIDEO{RESET}  {YELLOW}[{current_video}/{total_videos}]{RESET} {GREEN}{current_label[-78:]}{RESET}")
    print(f"{DIM}{str(current_path)[-112:]}{RESET}")
    print(f"\n{WHITE}PROGRESS{RESET}")
    print(f"  Stage: {stage_label:<10} [{_progress_bar(stage_percent, 100, 28)}] {stage_percent:3d}%")
    print(f"  Chunk: {chunk:<10} Remaining: {remaining_steps}")
    print(f"  Action: {activity}")
    print(f"  Speed: {speed}  Elapsed: {elapsed}")
    print(f"{WHITE}MODEL{RESET}     {model_status}" + (f"  {model_percent}" if model_percent else ""))
    if model_error:
        print(RED + f"MODEL ERROR  {model_error[-150:]}" + RESET)

    if batch_mode:
        print(f"\n{WHITE}QUEUE{RESET}")
        for index, path in enumerate(inventory[:10], 1):
            normalized = str(path).casefold()
            display_name = Path(path).name[-58:]
            if index <= completed_videos:
                marker, color, state = "DONE", GREEN, "DONE"
            elif index == current_video or normalized == str(current_path).casefold():
                marker, color, state = "RUN ", CYAN, "RUNNING"
            elif index <= completed_videos + failed_videos:
                marker, color, state = "FAIL", RED, "FAILED"
            else:
                marker, color, state = "WAIT", DIM, "WAITING"
            print(f"  {color}{marker}{RESET} {index:02d} {display_name:<58} {color}{state}{RESET}")
        if len(inventory) > 10:
            print(f"  {DIM}... and {len(inventory) - 10} more video(s){RESET}")

    print(f"\n{WHITE}RECENT LOG{RESET}")
    for line in list(recent)[-7:]:
        print(f"  {line[-118:]}")
    print(f"\n{DIM}Warnings {len(warnings)}  |  Errors {len(errors)}  |  {last[-130:]}{RESET}")
    print(f"{DIM}Keys: Ctrl+C stop safely  |  dashboard refreshes every second{RESET}")


def show_help():
    clear(); header()
    print("""
OPERATIONS
  Full       Transcribe + translate + dub.
  Split      Prepare chunks only; no dubbed media is produced.
  Transcribe Extract the source script and source-language SRT.
  Translate  Translate existing transcription; no TTS is run.
  Dub        Dub using existing translation files.

FINAL DELIVERABLES (depend on the selected operation)
  <source>_source.srt             Source-language transcript.
  <source>_normal.srt             Natural Arabic translation.
  <source>_fitted.srt             Timing-fitted Arabic translation.
  <source>_arabic.srt             Selected/actually used Arabic text.
  <source>_bilingual_en_ar.srt    English + Arabic cues; automatic for
                                  Translate, Dub, and Full (required output).
  <source>_dubbed.m4a             Dubbed audio.
  <source>_dubbed.mp4             Video with dubbed audio, if Output=video.
  dubbing_report.json/.txt        Timing, speed, and validation summary.

OTHER SETTINGS
  Local file, TXT list, or folder; optional recursive batch search.
  Source/target language, translation/Whisper models, CUDA/CPU,
  voice/reference audio, output type, silence-aware splitting, text mode,
  and per-sentence speed limit up to 2.5x.

Files are saved under the source-adjacent <source>_dubbing\\final folder.
Stop with Ctrl+C or close the console window with X; the active worker tree is stopped and completed checkpoints are kept.
Run the same input again to reuse completed stages/chunks and continue unfinished work.
Transcribe alone cannot make the bilingual SRT until translation is run.
""")
    pause()


def prompt_text(label, value=""):
    result=input(f"{label} [{value}]: ").strip()
    return result.strip('"') if result else value


def choose(label, options, current=0):
    clear(); header(); print(CYAN+"\n"+label+RESET+"\n")
    for i,option in enumerate(options,1): print(f"  {YELLOW}{i:2d}{RESET}. {option}")
    while True:
        ans=input(f"\nSelect number [{current+1}]: ").strip()
        idx=current if not ans else (int(ans)-1 if ans.isdigit() else -1)
        if 0 <= idx < len(options): return idx
        print(RED+"Invalid choice."+RESET)


def choose_profile(data, title):
    names=list(data)
    if not names:
        print("No saved profiles."); pause(); return None
    idx=choose(title,names)
    return names[idx]


def _set_option(s,key,options,current): s[key]=options[choose("Edit "+key, [x[0] if isinstance(x,tuple) else str(x) for x in options], current)]


def edit_reference_audio(s):
    """Choose, replace, or explicitly clear the reference voice file."""
    current = s.get("reference_audio") or ""
    clear(); header()
    print(CYAN+"\nReference audio"+RESET+"\n")
    print(f"  Current: {GREEN}{current or 'Default voice'}{RESET}")
    print(f"  {YELLOW}1{RESET}. Choose or replace reference audio")
    print(f"  {YELLOW}2{RESET}. Remove reference audio (use default voice)")
    print(f"  {YELLOW}3{RESET}. Keep current value")
    while True:
        answer = input("\nSelect number [1]: ").strip() or "1"
        if answer == "1":
            value = input("Reference audio path (type clear/none to remove): ").strip().strip('"')
            if value.lower() in {"clear", "none", "default", "reset", "-"}:
                s["reference_audio"] = ""
            elif value:
                s["reference_audio"] = value
            return
        if answer == "2":
            s["reference_audio"] = ""
            return
        if answer == "3":
            return
        print(RED+"Invalid choice."+RESET)


def edit(s, number):
    if number==1: s["input"]=prompt_text("Local media path, TXT list, or folder path",s["input"])
    elif number==2: s["source_mode"]=["single","list","folder"][choose("Input source type",["One local media file","TXT list (one path per line)","Folder containing media files"],["single","list","folder"].index(s["source_mode"]))]
    elif number==3: s["recursive"]=[False,True][choose("Search folder subdirectories?",["No (top folder only)","Yes (recursive)"],1 if s["recursive"] else 0)]
    elif number==4: s["source_language"]=LANGS[choose("Source language",[x[0] for x in LANGS],next((i for i,x in enumerate(LANGS) if x[1]==s["source_language"]),0))][1]
    elif number==5:
        x=TARGETS[choose("Target language",[x[0] for x in TARGETS],next((i for i,x in enumerate(TARGETS) if x[2]==s["target_code"]),0))]
        s["target_language"],s["target_code"]=x[1],x[2]
    elif number==6: s["llm"]=LLMS[choose("Translation model",[x[0] for x in LLMS],next((i for i,x in enumerate(LLMS) if x[1]==s["llm"]),0))][1]
    elif number==7: s["whisper"]=WHISPERS[choose("Whisper model",WHISPERS,WHISPERS.index(s["whisper"]) if s["whisper"] in WHISPERS else 0)]
    elif number==8: s["device"]=["cuda","cpu"][choose("Device",["CUDA GPU","CPU"],0 if s["device"]=="cuda" else 1)]
    elif number==9: s["gender"]=["Man","Woman"][choose("Voice",["Man","Woman"],0 if s["gender"]=="Man" else 1)]
    elif number==10: s["output"]=["video","audio"][choose("Output",["Video + audio","Audio only"],0 if s["output"]=="video" else 1)]
    elif number==11: edit_reference_audio(s)
    elif number==12: s["split_mode"]=["auto","never","always"][choose("Split mode",["Auto (threshold)","Never split","Always split"],["auto","never","always"].index(s["split_mode"]))]
    elif number==13:
        value=float(prompt_text("Maximum per-sentence speed (1.0-2.5)",str(s["max_speed"])))
        if not 1.0 <= value <= 2.5: raise ValueError("Maximum speed must be between 1.0 and 2.5")
        s["max_speed"]=value
    elif number==14: s["split_threshold_seconds"]=float(prompt_text("Split threshold in seconds",str(s["split_threshold_seconds"])))
    elif number==15: s["chunk_minutes"]=float(prompt_text("Chunk target in minutes",str(s["chunk_minutes"])))
    elif number==16: s["search_window"]=float(prompt_text("Silence search window in seconds",str(s["search_window"])))
    elif number==17: s["whisper_batch_size"]=[1,2,4,8][choose("Whisper batch size",["1 (lowest VRAM)","2","4 (safe default)","8"],[1,2,4,8].index(s["whisper_batch_size"]))]
    elif number==18: s["task"]=["full","split","transcribe","translate_srt","bilingual_srt","dub","srt_dub"][choose("Operation",["Full pipeline","Split only","Extract/transcribe source SRT","Output translated SRT","Output bilingual EN + target SRT","Dub using generated translation","Dub using external SRT"],["full","split","transcribe","translate_srt","bilingual_srt","dub","srt_dub"].index(s["task"]))]
    elif number==19: s["external_srt"]=prompt_text("External SRT path (used by SRT dubbing)",s.get("external_srt", ""))
    elif number==20: s["text_mode"]=["auto","normal","fitted"][choose("Text used for dubbing",["Auto: Normal, speed-up, then Fitted fallback","Normal only (default)","Fitted only"],["auto","normal","fitted"].index(s["text_mode"]))]
def render_menu(s):
    clear(); header()
    print(f"\n{WHITE}SETTINGS{RESET}  {DIM}(enter a number to edit a setting){RESET}\n")
    rows = [
        ("Input / batch source", s["input"] or "<not selected>"),
        ("Source type", s["source_mode"]),
        ("Search subfolders", "Yes" if s["recursive"] else "No"),
        ("Source language", s["source_language"]),
        ("Target language", f"{s['target_language']} ({s['target_code']})"),
        ("Translation model", s["llm"].split(" (")[0]),
        ("Whisper model", s["whisper"]),
        ("Device", s["device"]),
        ("Voice", s["gender"]),
        ("Output", s["output"]),
        ("Reference audio", s["reference_audio"] or "Default voice"),
        ("Split mode", s["split_mode"]),
        ("Max speed", f"{s['max_speed']}x"),
        ("Split threshold", f"{s['split_threshold_seconds']} sec"),
        ("Chunk target", f"{s['chunk_minutes']} min"),
        ("Silence search", f"{s['search_window']} sec"),
        ("Whisper batch size", str(s["whisper_batch_size"])),
        ("Operation", s["task"]),
        ("External SRT", s.get("external_srt") or "Not selected"),
        ("Dubbing text mode", s["text_mode"]),
    ]
    groups = (("INPUT", 0, 3), ("LANGUAGE AND MODELS", 3, 7), ("VOICE AND OUTPUT", 7, 11), ("PROCESSING", 11, 17), ("OPERATION", 17, 20))
    for title, start, end in groups:
        print(f"{CYAN}{title}{RESET}")
        print(f"{DIM}{'-' * 76}{RESET}")
        for i in range(start, end):
            label, value = rows[i]
            shown = str(value)
            if len(shown) > 48:
                shown = "..." + shown[-45:]
            print(f"  {YELLOW}{i + 1:2d}{RESET}. {label:<22} {GREEN}{shown}{RESET}")
        print()
    print(f"{DIM}SRT operations create translated and/or bilingual files in the final folder.{RESET}")
    print(f"\n{GREEN}S{RESET}. Start   {CYAN}H{RESET}. Help   {CYAN}P{RESET}. Save profile   {CYAN}L{RESET}. Load profile   {CYAN}D{RESET}. Delete profile   {RED}0{RESET}. Exit")


def run(s):
    if not s["input"]:
        print(RED+"A local input, TXT list, or folder path is required."+RESET); pause(); return
    src=Path(s["input"]).expanduser()
    inventory=input_inventory(s)
    if not inventory:
        print(RED+"No readable video/audio files were found in the selected input."+RESET); pause(); return
    workspace=src.parent/(src.stem+"_dubbing")
    log_path=workspace/"logs"/f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd=[sys.executable,str(BASE/"dub_cli.py")]
    if s["source_mode"]=="single": cmd += [str(src)]
    elif s["source_mode"]=="list": cmd += ["--batch-list",str(src)]
    else:
        cmd += ["--batch-dir",str(src)]
        if s["recursive"]: cmd.append("--recursive")
    cmd += ["--source-language",s["source_language"],"--target-language",s["target_language"],"--target-code",s["target_code"],"--llm",s["llm"],"--whisper",s["whisper"],"--whisper-batch-size",str(s["whisper_batch_size"]),"--device",s["device"],"--gender",s["gender"],"--output",s["output"],"--split-mode",s["split_mode"],"--max-speed",str(s["max_speed"]),"--split-threshold-seconds",str(s["split_threshold_seconds"]),"--chunk-minutes",str(s["chunk_minutes"]),"--search-window",str(s["search_window"]),"--task",s["task"],"--text-mode",s["text_mode"]]
    if s["source_mode"]=="single": cmd += ["--log-file",str(log_path)]
    if s["reference_audio"]: cmd += ["--reference-audio",s["reference_audio"]]
    if s.get("external_srt"): cmd += ["--external-srt",s["external_srt"]]
    check=subprocess.run([sys.executable,str(BASE/"dub_cli.py"),"--check","--whisper",s["whisper"]],cwd=BASE,text=True,encoding="utf-8",errors="replace",capture_output=True)
    check_text=(check.stdout or check.stderr).strip()
    destination_note=f"Workspace: {workspace}\nFinal output: {workspace/'final'}" if s["source_mode"]=="single" else "Batch mode: sequential; each input gets its own sibling <name>_dubbing\\final folder and run log."
    if s["source_mode"] != "single":
        destination_note += "\nBATCH ORDER: ONE VIDEO AT A TIME. Full finishes Transcribe -> Translate -> Dub for each video before the next starts."
    clear(); header()
    print(f"\n{WHITE}CONFIRM JOB{RESET}  {DIM}(review before starting){RESET}\n")
    print(f"{CYAN}INPUT INVENTORY{RESET}")
    print(f"{DIM}{'-' * 76}{RESET}")
    print(f"Files to process: {GREEN}{len(inventory)}{RESET}   Source type: {GREEN}{s['source_mode']}{RESET}   Recursive: {GREEN}{'Yes' if s['recursive'] else 'No'}{RESET}")
    for index,path in enumerate(inventory[:8],1):
        print(f"  {index:02d}. {str(path)[-102:]}")
    if len(inventory)>8: print(f"  {DIM}... and {len(inventory)-8} more file(s){RESET}")
    print(f"\n{CYAN}RUN SETTINGS{RESET}")
    print(f"{DIM}{'-' * 76}{RESET}")
    print(f"Operation: {GREEN}{s['task']}{RESET}   Output: {GREEN}{s['output']}{RESET}   Device: {GREEN}{s['device']}{RESET}")
    print(f"Language: {GREEN}{s['source_language']} -> {s['target_language']} ({s['target_code']}){RESET}")
    print(f"Models: Whisper={GREEN}{s['whisper']}{RESET}  Translation={GREEN}{s['llm'].split(' (')[0]}{RESET}")
    print(f"Voice: {GREEN}{s['gender']}{RESET}  Reference: {GREEN}{Path(s['reference_audio']).name if s['reference_audio'] else 'Default voice'}{RESET}")
    print(f"Workspace: {workspace}\n")
    print(f"{CYAN}ENVIRONMENT CHECK{RESET}")
    print(f"{DIM}{'-' * 76}{RESET}")
    for line in check_text.splitlines()[-12:]:
        print(f"  {line[-102:]}")
    print(f"\n{CYAN}COMMAND{RESET}\n{DIM}{subprocess.list2cmdline(cmd)}{RESET}\n")
    print(f"{CYAN}OUTPUT{RESET}\n{destination_note}")
    print(f"\n{GREEN}Press Enter or type Y to start{RESET}   {YELLOW}N{RESET} Cancel")
    if input("\nStart job? [Y/n]: ").strip().lower() not in {"", "y", "yes"}: return
    try:
        temporary = LAST_SESSION_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, LAST_SESSION_FILE)
    except OSError as exc:
        print(f"[WARN] Could not save last session settings: {exc}")
    env=os.environ.copy(); env["PYTHONUNBUFFERED"]="1"; env["PYTHONIOENCODING"]="utf-8:backslashreplace"
    output=queue.Queue(); start=time.monotonic()
    proc=subprocess.Popen(cmd,cwd=BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
    def on_console_close(control_type):
        if int(control_type) in (2,5,6):
            terminate_process_tree(proc, timeout=2.0)
            return True
        return False
    console_handler = install_console_close_handler(on_console_close)
    def reader():
        try:
            for line in proc.stdout: output.put(line.rstrip("\r\n"))
        finally: output.put(None)
    threading.Thread(target=reader,daemon=True).start()
    recent=deque(maxlen=16); warnings=deque(maxlen=100); errors=deque(maxlen=100); last="Starting..."; stage="Starting"; chunk="-"; done=False; next_render=0.0
    total_videos=len(inventory); completed_videos=0; failed_videos=0; current_video=1; current_path=str(inventory[0])
    batch_mode = s["source_mode"] in {"folder", "list"}
    stage_percent=0; speed="unknown"; remaining_steps="starting"; model_status="No model activity yet"; model_percent=""; model_error=""; activity="Preparing input"
    last_extract_at=None; extract_speed=""
    warning_traceback=False; collecting_failure=False
    try:
        while not done or proc.poll() is None:
            changed=False
            try:
                first=output.get(timeout=0.25)
                pending=[first]
            except queue.Empty:
                pending=[]
            while True:
                try: pending.append(output.get_nowait())
                except queue.Empty: break
            for item in pending:
                if item is None:
                    done=True; changed=True; continue
                changed=True; last=item; recent.append(item); low=item.lower()
                progress_match=re.search(r"\[progress\]\s+videos_total=(\d+)\s+videos_completed=(\d+)\s+videos_failed=(\d+)\s+current_video=(\d+)\s+current_path=(.*?)\s+stage=(\S+)\s+stage_percent=(\d+)\s+speed=(.*?)\s+remaining=(.*?)(?:\s+error=(.*))?$", item, re.IGNORECASE)
                if progress_match:
                    progress_total=int(progress_match.group(1))
                    # In folder/list mode the outer CLI owns the batch counters.
                    # Each child video process emits videos_total=1; accepting that
                    # value here makes the TUI incorrectly display 1/N as 1 total.
                    if batch_mode and total_videos > 1 and progress_total == 1:
                        current_path=progress_match.group(5); stage=progress_match.group(6); stage_percent=int(progress_match.group(7)); speed=progress_match.group(8)
                        remaining_steps=progress_match.group(9); model_error=progress_match.group(10) or model_error
                    else:
                        total_videos=progress_total; completed_videos=int(progress_match.group(2)); failed_videos=int(progress_match.group(3)); current_video=int(progress_match.group(4)); current_path=progress_match.group(5); stage=progress_match.group(6); stage_percent=int(progress_match.group(7)); speed=progress_match.group(8); remaining_steps=progress_match.group(9); model_error=progress_match.group(10) or model_error
                percent_match=re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", item)
                speed_match=re.search(r"(\d+(?:\.\d+)?)\s*(it/s|steps?/s|[kmgt]?b/s|samples?/s)", item, re.IGNORECASE)
                if percent_match and not progress_match: stage_percent=min(100, int(float(percent_match.group(1))))
                if speed_match and not progress_match: speed=speed_match.group(1)+" "+speed_match.group(2)
                segment_match=re.search(r"(\d+)\s*/\s*(\d+)\s+segments translated", low)
                if segment_match and not progress_match:
                    segment_done=int(segment_match.group(1)); segment_total=max(1, int(segment_match.group(2)))
                    stage_percent=min(100, int(segment_done * 100 / segment_total))
                    remaining_steps=f"{max(0, segment_total-segment_done)} segment(s) in current chunk"
                if any(token in low for token in ("download", "downloading", "loading model", "load model", "model download")):
                    model_status=item[-260:]
                    if percent_match: model_percent=percent_match.group(1)+"%"
                if any(token in low for token in ("model", "whisper", "torch", "cuda", "checkpoint")) and any(token in low for token in ("error", "failed", "exception", "not found", "missing")):
                    model_error=item[-300:]
                if "[resume]" in low: activity="Reusing checkpoints and jumping to the first incomplete chunk"
                elif "reading" in low or "probe" in low or "duration" in low: activity="Reading file metadata and duration"
                elif "quiet gap" in low or "silence" in low: activity="Analyzing silence and split points"
                elif "extracting audio" in low: activity="Extracting audio chunk"
                elif "loading llm" in low or "load_model" in low: activity="Loading translation model"
                elif "transcrib" in low: activity="Transcribing speech"
                elif "translat" in low: activity="Translating text"
                elif "separat" in low: activity="Separating vocals and background"
                elif "dubb" in low: activity="Generating dubbed speech"
                elif "assembl" in low or "ffmpeg" in low: activity="Assembling final media"
                extract_match=re.search(r"extracting audio.*?:\s*([0-9.]+)s\s*->\s*([0-9.]+)s", low)
                if extract_match:
                    wall_now=time.monotonic()
                    if last_extract_at is not None:
                        media_seconds=abs(float(extract_match.group(2))-float(extract_match.group(1)))
                        wall_seconds=max(0.01, wall_now-last_extract_at)
                        extract_speed=f"{media_seconds/wall_seconds:.1f}x media-time"
                    last_extract_at=wall_now
                    speed=extract_speed or "measuring..."
                classification=classify_log_line(item,warning_traceback,collecting_failure)
                warning_traceback=classification["warning_traceback"]
                collecting_failure=classification["collecting_failure"]
                if classification["warning"]: warnings.append(item)
                if classification["error"]: errors.append(item)
                for token,label in [("split","Split"),("transcrib","Transcription"),("translat","Translation"),("separat","Separation"),("dub","Dubbing"),("assembl","Assembly")]:
                    if token in low: stage=label
                m=re.search(r"chunk[_ ]?(\d+)(?:\s*/\s*(\d+))?",low)
                if m:
                    chunk=(m.group(1)+("/"+m.group(2) if m.group(2) else ""))
                    if not progress_match and m.group(2):
                        chunk_index=int(m.group(1)); chunk_total=max(1, int(m.group(2)))
                        stage_percent=min(100, int(chunk_index * 100 / chunk_total))
                        remaining_steps=f"{max(0, chunk_total-chunk_index)} chunk(s)"
            now=time.monotonic()
            if now>=next_render or (done and proc.poll() is not None):
                render_dashboard(
                    total_videos=total_videos, completed_videos=completed_videos,
                    failed_videos=failed_videos, current_video=current_video,
                    current_path=current_path, inventory=inventory, stage=stage,
                    stage_percent=stage_percent, chunk=chunk,
                    remaining_steps=remaining_steps, activity=activity,
                    speed=speed, elapsed=timedelta(seconds=int(now-start)),
                    model_status=model_status, model_percent=model_percent,
                    model_error=model_error, warnings=warnings, errors=errors,
                    last=last, recent=recent, batch_mode=batch_mode,
                )
                log_display = str(log_path) if s["source_mode"] == "single" else "Each video has its own log under <video>_dubbing\\logs"
                print(DIM+f"Full log: {log_display}"+RESET)
                next_render=now+1.0
        code=proc.wait()
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received. Stopping the CLI and its active dubbing/FFmpeg descendants; completed checkpoints are preserved.")
        stopped = terminate_process_tree(proc)
        code = 130
        if stopped:
            print("[STOP] Worker tree stopped. Re-run the same source to resume.")
        else:
            print("[STOP WARNING] Worker-tree shutdown could not be confirmed; its run lock will prevent a duplicate job.")
    finally:
        remove_console_close_handler(console_handler)
    if not log_path.exists():
        log_path.parent.mkdir(parents=True,exist_ok=True)
        log_path.write_text("COMMAND: "+subprocess.list2cmdline(cmd)+"\n"+"\n".join(recent)+"\n",encoding="utf-8")
    status_label = YELLOW+"STOPPED — CHECKPOINTS KEPT" if code==130 else (GREEN+"COMPLETED" if code==0 else RED+"FAILED")
    clear(); header(); print(status_label+RESET+f"\nElapsed: {timedelta(seconds=int(time.monotonic()-start))}\nLog: {log_path}")
    if code and errors: print(RED+"\nRecent error/traceback lines:\n"+"\n".join(list(errors)[-30:])+RESET)
    if code==0: print(f"\nFinal outputs: {workspace/'final'}")
    pause()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try: stream.reconfigure(errors="backslashreplace")
            except (OSError, ValueError): pass
    s=defaults()
    while True:
        render_menu(s); choice=input("\nSelect option (number / S / H / P / L / D / 0): ").strip().lower()
        if choice in {"0","q","quit"}: return
        if choice=="s": run(s); continue
        if choice=="h": show_help(); continue
        if choice=="p":
            name=prompt_text("Profile name")
            if name:
                data=profiles(); data[name]={k:v for k,v in s.items() if k not in {"input","cps"}}; save_profiles(data); print("Profile saved."); pause()
            continue
        if choice=="l":
            data=profiles(); name=choose_profile(data,"Load profile")
            if name: s.update({k:v for k,v in data[name].items() if k!="cps"}); s["input"]=""
            continue
        if choice=="d":
            data=profiles(); name=choose_profile(data,"Delete profile")
            if name and input(f"Delete profile '{name}'? [y/N]: ").strip().lower()=="y": data.pop(name,None); save_profiles(data)
            continue
        if choice.isdigit() and 1<=int(choice)<=20:
            try: edit(s,int(choice))
            except (ValueError,IndexError) as exc: print(RED+str(exc)+RESET); pause()


if __name__=="__main__": main()
