import os
import re
import subprocess
from pathlib import Path

import soundfile as sf

from config import OUTPUT_SAMPLE_RATE


class AudioMixer:
    """Disk-streaming voice/background mixer.

    The original behavior is preserved: mono inputs are upmixed to stereo,
    short backgrounds repeat, long backgrounds are truncated to the voice
    duration, gains are applied, and the combined signal is globally attenuated
    only if its peak would exceed full scale. FFmpeg keeps audio frames in
    bounded buffers instead of decoding both full tracks into NumPy arrays.
    """

    def mix(self, voice_path, background_path, output_path, voice_volume_db=0, bg_volume_db=-6):
        voice_path = Path(voice_path)
        background_path = Path(background_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice_info = sf.info(str(voice_path))
        if voice_info.frames <= 0 or voice_info.samplerate <= 0:
            raise ValueError(f"Voice track is empty or invalid: {voice_path}")
        duration = voice_info.frames / voice_info.samplerate

        filters = (
            f"[0:a]aresample={OUTPUT_SAMPLE_RATE},aformat=channel_layouts=stereo,"
            f"volume={voice_volume_db}dB[voice];"
            f"[1:a]aresample={OUTPUT_SAMPLE_RATE},aformat=channel_layouts=stereo,"
            f"volume={bg_volume_db}dB[background];"
            "[voice][background]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0,"
            f"atrim=duration={duration:.9f}"
        )

        def run_pass(*, detect_peak: bool, correction_db: float = 0.0):
            tail = ",astats=metadata=0:reset=0[mixed]" if detect_peak else f",volume={correction_db:.8f}dB[mixed]"
            command = [
                "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "info",
                "-i", str(voice_path), "-stream_loop", "-1", "-i", str(background_path),
                "-filter_complex", filters + tail, "-map", "[mixed]", "-t", f"{duration:.9f}",
                "-vn", "-sn", "-dn",
            ]
            if detect_peak:
                command += ["-f", "null", "-"]
            else:
                command += ["-ar", str(OUTPUT_SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(output_path)]
            return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace", check=False)

        analysis = run_pass(detect_peak=True)
        if analysis.returncode != 0:
            raise RuntimeError(f"FFmpeg peak analysis failed:\n{analysis.stderr[-3000:]}")

        peaks = [float(value) for value in re.findall(r"Peak level dB:\s*(-?\d+(?:\.\d+)?)", analysis.stderr, re.IGNORECASE)]
        correction_db = 0.0
        if peaks:
            peak_db = max(peaks)
            if peak_db > 0.0:
                correction_db = -peak_db

        rendered = run_pass(detect_peak=False, correction_db=correction_db)
        if rendered.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"FFmpeg audio mixing failed:\n{rendered.stderr[-3000:]}")
        return str(output_path)

    def export_for_youtube(self, mixed_path, output_dir, basename):
        """Export as WAV and MP3."""
        wav_path = os.path.join(output_dir, f"{basename}_audio.wav")
        mp3_path = os.path.join(output_dir, f"{basename}_audio.mp3")

        import shutil
        shutil.copy2(mixed_path, wav_path)

        cmd = [
            "ffmpeg", "-nostdin", "-y", "-i", mixed_path,
            "-b:a", "320k", mp3_path,
        ]
        subprocess.run(cmd, check=True)

        return {"wav": wav_path, "mp3": mp3_path}
