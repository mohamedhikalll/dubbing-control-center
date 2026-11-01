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


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def profiles():
    try:
        value=json.loads(PROFILE_FILE.read_text(encoding="utf-8")); return value if isinstance(value,dict) else {}
    except (OSError,json.JSONDecodeError): return {}


def save_profiles(data): PROFILE_FILE.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def defaults():
    settings = {"input":"","source_mode":"single","recursive":False,"source_language":"auto","target_language":"Arabic","target_code":"ar","llm":LLMS[0][1],"whisper":"large-v3","whisper_batch_size":4,"device":"cuda","gender":"Man","output":"video","reference_audio":"","split_mode":"auto","chunk_minutes":10.0,"split_threshold_seconds":1140.0,"search_window":30.0,"max_speed":2.5,"task":"full","text_mode":"normal"}
    try:
        saved = json.loads(LAST_SESSION_FILE.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            settings.update({key: value for key, value in saved.items() if key in settings})
    except (OSError, json.JSONDecodeError):
        pass
    return settings


def header():
    print(CYAN+"╔"+"═"*68+"╗\n║"+GREEN+"                  DUBBING CONTROL CENTER                  "+RESET+CYAN+"║\n╚"+"═"*68+"╝"+RESET)


def pause(): input("\nPress Enter to continue...")


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
    elif number==11: s["reference_audio"]=prompt_text("Reference audio path (blank = default voice)",s["reference_audio"] or "")
    elif number==12: s["split_mode"]=["auto","never","always"][choose("Split mode",["Auto (threshold)","Never split","Always split"],["auto","never","always"].index(s["split_mode"]))]
    elif number==13:
        value=float(prompt_text("Maximum per-sentence speed (1.0-2.5)",str(s["max_speed"])))
        if not 1.0 <= value <= 2.5: raise ValueError("Maximum speed must be between 1.0 and 2.5")
        s["max_speed"]=value
    elif number==14: s["split_threshold_seconds"]=float(prompt_text("Split threshold in seconds",str(s["split_threshold_seconds"])))
    elif number==15: s["chunk_minutes"]=float(prompt_text("Chunk target in minutes",str(s["chunk_minutes"])))
    elif number==16: s["search_window"]=float(prompt_text("Silence search window in seconds",str(s["search_window"])))
    elif number==17: s["whisper_batch_size"]=[1,2,4,8][choose("Whisper batch size",["1 (lowest VRAM)","2","4 (safe default)","8"],[1,2,4,8].index(s["whisper_batch_size"]))]
    elif number==18: s["task"]=["full","split","transcribe","translate","dub"][choose("Operation",["Full pipeline","Split only","Extract/transcribe script only","Translate existing transcription only","Dub using existing translation only"],["full","split","transcribe","translate","dub"].index(s["task"]))]
    elif number==19: s["text_mode"]=["auto","normal","fitted"][choose("Text used for dubbing",["Auto: Normal, speed-up, then Fitted fallback","Normal only (default)","Fitted only"],["auto","normal","fitted"].index(s["text_mode"]))]
def render_menu(s):
    clear(); header(); print("\n"+WHITE+"Current settings"+RESET+"\n")
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
        ("Dubbing text mode", s["text_mode"]),
    ]
    for i,(label,value) in enumerate(rows,1): print(f"  {YELLOW}{i:2d}{RESET}. {label:<22} {GREEN}{value}{RESET}")
    print(f"  {DIM}Automatic required output: <source>_bilingual_en_ar.srt (EN + Arabic) for translate/dub/full.{RESET}")
    print(f"\n  {GREEN}S{RESET}. Start selected operation\n  {CYAN}H{RESET}. Show operations and outputs\n  {CYAN}P{RESET}. Save profile\n  {CYAN}L{RESET}. Load profile\n  {CYAN}D{RESET}. Delete profile\n  {RED}0{RESET}. Exit")


def run(s):
    if not s["input"]:
        print(RED+"A local input, TXT list, or folder path is required."+RESET); pause(); return
    src=Path(s["input"]).expanduser()
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
    check=subprocess.run([sys.executable,str(BASE/"dub_cli.py"),"--check","--whisper",s["whisper"]],cwd=BASE,text=True,encoding="utf-8",errors="replace",capture_output=True)
    check_text=(check.stdout or check.stderr).strip()
    destination_note=f"Workspace: {workspace}\nFinal output: {workspace/'final'}" if s["source_mode"]=="single" else "Batch mode: sequential; each input gets its own sibling <name>_dubbing\\final folder and run log."
    if s["source_mode"] != "single":
        destination_note += "\nBATCH ORDER: ONE VIDEO AT A TIME. Full finishes Transcribe -> Translate -> Dub for each video before the next starts."
    clear(); header(); print("\nENVIRONMENT CHECK\n"+check_text+"\n\nCONFIRM JOB\n\nCommand:\n"+subprocess.list2cmdline(cmd)+f"\n\n{destination_note}")
    if input("\nStart? [Y/n]: ").strip().lower() not in {"", "y", "yes"}: return
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
                classification=classify_log_line(item,warning_traceback,collecting_failure)
                warning_traceback=classification["warning_traceback"]
                collecting_failure=classification["collecting_failure"]
                if classification["warning"]: warnings.append(item)
                if classification["error"]: errors.append(item)
                for token,label in [("split","Split"),("transcrib","Transcription"),("translat","Translation"),("separat","Separation"),("dub","Dubbing"),("assembl","Assembly")]:
                    if token in low: stage=label
                m=re.search(r"chunk[_ ]?(\d+)(?:\s*/\s*(\d+))?",low)
                if m: chunk=(m.group(1)+("/"+m.group(2) if m.group(2) else ""))
            now=time.monotonic()
            if now>=next_render or (done and proc.poll() is not None):
                clear(); header(); print(f"\nCurrent stage: {stage}\nCurrent chunk: {chunk}\nElapsed: {timedelta(seconds=int(now-start))}\nWarnings: {len(warnings)}"+(f" | {warnings[-1][-180:]}" if warnings else "")+f"\nErrors: {len(errors)}"+(f" | {errors[-1][-180:]}" if errors else "")+f"\nLast message: {last[-250:]}\n\nRecent log:")
                for line in recent: print(line[-240:])
                print(DIM+f"\nFull log: {log_path}"+RESET)
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
        if choice.isdigit() and 1<=int(choice)<=19:
            try: edit(s,int(choice))
            except (ValueError,IndexError) as exc: print(RED+str(exc)+RESET); pause()


if __name__=="__main__": main()
