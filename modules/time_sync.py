import numpy as np
import soundfile as sf
import librosa
import hashlib
import json
import os
from modules.utils import get_exact_duration
from config import (
    TOLERANCE_TOO_LONG, MIN_SEGMENT_DURATION, TEMP_DIR,
    OUTPUT_SAMPLE_RATE, CHARS_PER_SECOND, MAX_SPEED_FACTOR,
    NEVER_CUT_WARNING, GPU_VRAM_GB
)

# Two-pass: natural voice first, native speed instruction if overflow

class TimeSync:
    def __init__(self, tts_engine, reformulator=None):
        self.tts = tts_engine
        self.reformulator = reformulator

    def _audio_cache_tag(self, text, language, start, end, voice_path, gender, speed_cap):
        cache_voice = voice_path
        if not cache_voice:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_voice = os.path.join(project_root, "voices", f"default_{str(gender).lower()}.wav")
        voice_identity = {"path": os.path.abspath(str(cache_voice)) if cache_voice else None}
        if cache_voice and os.path.isfile(cache_voice):
            voice_stat = os.stat(cache_voice)
            voice_identity.update(size=voice_stat.st_size, mtime_ns=voice_stat.st_mtime_ns)
        identity = json.dumps({
            "text": str(text), "language": str(language or "all"),
            "start": float(start), "end": float(end), "gender": str(gender),
            "voice": voice_identity,
            "engine": f"{type(self.tts).__module__}.{type(self.tts).__qualname__}",
            "sample_rate": OUTPUT_SAMPLE_RATE, "speed_cap": float(speed_cap),
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(identity).hexdigest()[:16]

    def _resample_to_target(self, audio, orig_sr, target_sr=OUTPUT_SAMPLE_RATE):
        """Resample audio to target sample rate if needed."""
        if orig_sr == target_sr:
            return audio, target_sr
        audio_resampled = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        return audio_resampled, target_sr



    @staticmethod
    def estimate_duration(text, language_family="latin"):
        """
        Estimate TTS duration from text length using chars/second ratios.
        Returns estimated duration in seconds.
        """
        cps = CHARS_PER_SECOND.get(language_family, CHARS_PER_SECOND["default"])
        return len(text) / cps

    @staticmethod
    def get_language_family(lang_code):
        """Map language codes to language families for duration estimation."""
        cjk = ["jpn", "zho", "kor", "ja", "zh", "ko"]
        arabic = ["arb", "ar"]
        for prefix in cjk:
            if lang_code.startswith(prefix):
                return "cjk"
        for prefix in arabic:
            if lang_code.startswith(prefix):
                return "arabic"
        return "latin"

    def _compute_effective_durations(self, segments, total_duration=None):
        """
        Compute effective duration for each segment using up to 2 gaps.
        
        1-gap: extend into the gap between this segment and the next.
        2-gap: if the NEXT segment also has slack (its gap > its strict duration),
               borrow that slack so the current segment has more room.
        
        This avoids truncating dense segments when followed by sparse ones.
        """
        GAP_MARGIN = 0.03  # 30ms safety margin per boundary
        
        for i, seg in enumerate(segments):
            strict_duration = seg["end"] - seg["start"]
            
            # 1-gap: space to next segment
            if i + 1 < len(segments):
                gap_1 = segments[i + 1]["start"] - seg["start"] - GAP_MARGIN
            elif total_duration:
                gap_1 = total_duration - seg["start"]
            else:
                gap_1 = strict_duration
            
            available = gap_1
            
            # 2-gap: borrow slack from next segment's gap
            # If next segment has more space than it needs, we can use the extra
            if i + 1 < len(segments) and i + 2 < len(segments):
                next_strict = segments[i + 1]["end"] - segments[i + 1]["start"]
                next_gap = segments[i + 2]["start"] - segments[i + 1]["start"]
                next_slack = max(0, next_gap - next_strict - GAP_MARGIN)
                if next_slack > 0:
                    available = gap_1 + next_slack
            elif i + 1 < len(segments) and total_duration:
                # Last pair: next segment can extend to total_duration
                next_strict = segments[i + 1]["end"] - segments[i + 1]["start"]
                next_gap = total_duration - segments[i + 1]["start"]
                next_slack = max(0, next_gap - next_strict - GAP_MARGIN)
                if next_slack > 0:
                    available = gap_1 + next_slack
            
            seg["effective_duration"] = max(strict_duration, available)
            seg["strict_duration"] = strict_duration

    def pre_check_and_shorten(self, segments, language):
        """
        Pre-check timing with GLOBAL gap awareness.
        Computes effective durations and logs overflow warnings.
        Text fitting is handled upstream by translator + reformulator.
        """
        lang_family = self.get_language_family(language)
        
        # Global pass: compute effective durations
        self._compute_effective_durations(segments)
        
        warning_count = 0
        for seg in segments:
            text = seg.get("translated_text", seg.get("text", ""))
            effective = seg["effective_duration"]
            estimated_duration = self.estimate_duration(text, lang_family)
            
            seg["estimated_duration"] = round(estimated_duration, 2)
            seg["pre_shortened"] = False
            
            # Log warnings for segments that may still overflow
            if estimated_duration > effective and effective > MIN_SEGMENT_DURATION:
                overflow_pct = (estimated_duration - effective) / effective * 100
                print(f"[WARN] Segment [{seg['start']:.1f}-{seg['end']:.1f}]: ~{overflow_pct:.0f}% over, may overflow")
                warning_count += 1
        
        if warning_count > 0:
            print(f"{warning_count} segments may still overflow (LLM reformulation will handle during TTS)")
        
        return segments, 0

    def _speed_audio_file(self, input_path, output_path, speed):
        """Speed up generated speech without changing pitch."""
        import subprocess

        speed = max(1.0, min(float(speed), float(MAX_SPEED_FACTOR)))
        remaining = speed
        filters = []
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        filters.append(f"atempo={remaining:.6f}")

        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-filter:a",
                ",".join(filters),
                "-ar",
                str(OUTPUT_SAMPLE_RATE),
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
        )

    def sync_segment(self, segment, language, voice_path=None, gender="Woman"):
        """
        Synchronize a segment with two-pass speed control:
        1. Generate TTS at natural speed
        2. If overflow detected, re-generate with native speed instruction
        3. If still overflows → truncate with fade-out to prevent overlap
        """
        strict_duration = segment.get("strict_duration", segment["end"] - segment["start"])
        
        if strict_duration < MIN_SEGMENT_DURATION:
            return {
                "synced_path": None,
                "start": segment["start"],
                "end": segment["end"],
                "final_text": "",
                "sped_up": False,
                "truncated": False,
            }

        text = segment.get("translated_text", segment.get("text", ""))
        effective_duration = segment.get("effective_duration", strict_duration)
        
        # Cache reuse check per language
        lang_tag = str(language).lower().strip()[:3] if language else "all"
        cache_tag = self._audio_cache_tag(
            text, language, segment["start"], segment["end"],
            voice_path, gender, MAX_SPEED_FACTOR,
        )
        speed_tag = f"{float(MAX_SPEED_FACTOR):.2f}".replace(".", "_")
        final_synced_path = os.path.join(TEMP_DIR, f"seg_{lang_tag}_{segment['start']:.2f}_{cache_tag}_cap{speed_tag}_synced.wav")
        if os.path.exists(final_synced_path):
            print(f"Segment [{segment['start']:.1f}-{segment['end']:.1f}] ({lang_tag}): Reusing cached audio {final_synced_path}")
            try:
                cached_audio, _cached_sr = sf.read(final_synced_path, dtype="float32", always_2d=False)
                if (cached_audio.size == 0 or not np.isfinite(cached_audio).all()
                        or float(np.max(np.abs(cached_audio))) < 1e-5):
                    raise ValueError("cached TTS audio is empty or effectively silent")
                del cached_audio
                info = sf.info(final_synced_path)
                actual_duration = info.duration
                metadata_path = final_synced_path + ".meta.json"
                if not os.path.isfile(metadata_path):
                    raise ValueError("cached TTS metadata is missing; checkpoint is incomplete")
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                if not isinstance(metadata, dict) or "speed_factor" not in metadata or "truncated" not in metadata:
                    raise ValueError("cached TTS metadata is invalid; checkpoint is incomplete")
                return {
                    "synced_path": final_synced_path,
                    "start": segment["start"],
                    "end": segment["end"],
                    "final_text": text,
                    "sped_up": bool(metadata.get("sped_up", False)),
                    "speed_factor": float(metadata.get("speed_factor", 1.0)),
                    "truncated": bool(metadata.get("truncated", False)),
                    "final_duration": round(actual_duration, 3),
                    "slot_duration": round(effective_duration, 3),
                    "overflow": 0.0,
                }
            except Exception as e:
                print(f"Failed to read cached segment audio info: {e}. Re-generating...")
                for stale in (final_synced_path, final_synced_path + ".meta.json", final_synced_path + ".writing.wav", final_synced_path + ".meta.json.tmp"):
                    try:
                        if os.path.isfile(stale):
                            os.unlink(stale)
                    except OSError:
                        pass

        was_sped_up = False
        speed_factor = 1.0
        was_truncated = False
        final_text = text
        # Check if backend supports duration control natively
        has_duration_control = self.tts.capabilities.get("duration_control", False)
        temp_tts_path = os.path.join(TEMP_DIR, f"seg_{lang_tag}_{segment['start']:.2f}_temp.wav")
        
        final_synced_path = os.path.join(TEMP_DIR, f"seg_{lang_tag}_{segment['start']:.2f}_{cache_tag}_cap{speed_tag}_synced.wav")
        try:
            if has_duration_control and effective_duration > MIN_SEGMENT_DURATION:
                # PASS 1: Generate EXACTLY at the target duration
                res = self.tts.generate(
                    text=text, language=language, output_path=temp_tts_path, 
                    ref_audio_path=voice_path, duration=effective_duration, gender=gender
                )
                current_duration = res["duration"]
                print(f"Segment [{segment['start']:.1f}-{segment['end']:.1f}] ({lang_tag}): Backend native duration control used ({current_duration:.2f}s)")
            else:
                # PASS 1: Generate TTS at natural speed (pass duration as hint for max_new_tokens capping)
                res = self.tts.generate(
                    text=text, language=language, output_path=temp_tts_path,
                    ref_audio_path=voice_path, duration=effective_duration, gender=gender
                )
                current_duration = res["duration"]

                overflow = current_duration - effective_duration

                # Real per-segment speed correction, capped by MAX_SPEED_FACTOR.
                # If the cap is not enough, Normal-only mode preserves text and
                # the final safety trim is reported instead of silent fitted fallback.
                if (
                    overflow > TOLERANCE_TOO_LONG
                    and effective_duration > MIN_SEGMENT_DURATION
                ):
                    required_speed = current_duration / effective_duration

                    if required_speed <= MAX_SPEED_FACTOR:
                        speed_path = os.path.join(
                            TEMP_DIR,
                            f"seg_{lang_tag}_{segment['start']:.2f}_speed.wav",
                        )

                        print(
                            f"Segment [{segment['start']:.1f}-{segment['end']:.1f}] "
                            f"speeding audio {current_duration:.2f}s -> "
                            f"{effective_duration:.2f}s "
                            f"({required_speed:.2f}x)"
                        )

                        self._speed_audio_file(
                            temp_tts_path,
                            speed_path,
                            required_speed,
                        )

                        temp_tts_path = speed_path
                        current_duration = effective_duration
                        overflow = 0.0
                        was_sped_up = True
                        speed_factor = required_speed
                final_text = text

                # PASS 2+: LLM reformulation loop if audio is too long (SPEC §N.2 — max 3 attempts)
                # Shield: on <= 8.5 GB GPUs, avoid reloading heavy LLM while TTS model is active in VRAM to prevent system RAM spillover
                can_use_llm = self.reformulator is not None
                if can_use_llm and GPU_VRAM_GB > 0 and GPU_VRAM_GB <= 8.5:
                    if getattr(self.reformulator, "llm", None) is None and getattr(self.tts, "model", None) is not None:
                        can_use_llm = False
                        if not getattr(self, "_vram_shield_warned", False):
                            print(f"[VRAM Shield] GPU ≤ 8.5 GB ({GPU_VRAM_GB:.1f} GB): skipping live LLM reload during TTS to prevent system RAM spillover. Relying on gap absorption & crossfade.")
                            self._vram_shield_warned = True

                if overflow > TOLERANCE_TOO_LONG and effective_duration > MIN_SEGMENT_DURATION and can_use_llm:
                    current_text = text
                    for attempt in range(3):
                        # Scale target chars proportionally to the audio overflow ratio + 10% safety
                        # This works for any TTS backend regardless of speaking rate
                        scale = (effective_duration / current_duration) * 0.90
                        target_chars = max(10, int(len(current_text) * scale))
                        print(f"Segment [{segment['start']:.1f}-{segment['end']:.1f}] ({lang_tag}): {overflow:.2f}s over -> LLM reformulation {attempt+1}/3 (<={target_chars} chars)")
                        try:
                            shorter_text = self.reformulator.shorten(current_text, target_chars, language)
                        except Exception as e:
                            print(f"  Reformulation failed: {e}")
                            break
                        if not shorter_text or len(shorter_text) >= len(current_text):
                            print(f"  Reformulation returned same/longer text, stopping")
                            break
                        temp_regen_path = os.path.join(TEMP_DIR, f"seg_{lang_tag}_{segment['start']:.2f}_regen{attempt}.wav")
                        res_regen = self.tts.generate(
                            text=shorter_text, language=language, output_path=temp_regen_path,
                            ref_audio_path=voice_path, duration=effective_duration, gender=gender
                        )
                        regen_overflow = res_regen["duration"] - effective_duration
                        print(f"  Regen: {res_regen['duration']:.2f}s (overflow: {regen_overflow:+.2f}s)")
                        # Keep the best (shortest that fits or closest to fitting)
                        if res_regen["duration"] < current_duration:
                            res = res_regen
                            temp_tts_path = temp_regen_path
                            current_duration = res_regen["duration"]
                            overflow = regen_overflow
                            final_text = shorter_text
                        current_text = shorter_text
                        if regen_overflow <= TOLERANCE_TOO_LONG:
                            print(f"  Reformulation OK: {current_duration:.2f}s fits in {effective_duration:.2f}s slot")
                            break
                    if overflow > TOLERANCE_TOO_LONG:
                        print(f"  [{segment['start']:.1f}-{segment['end']:.1f}] Still {overflow:.2f}s over after reformulation — accepting (crossfade will handle)")
                elif current_duration > strict_duration + TOLERANCE_TOO_LONG:
                    gap_used = current_duration - strict_duration
                    print(f"Segment [{segment['start']:.1f}-{segment['end']:.1f}] ({lang_tag}): using {gap_used:.2f}s from gap")

            # Update segment text directly if it was reformulated
            if final_text != text:
                segment["translated_text"] = final_text
                if "fitted_text" in segment:
                    segment["fitted_text"] = final_text
                if "normal_text" in segment:
                    segment["normal_text"] = final_text

            # READ, RESAMPLE, SAVE
            audio, sr = sf.read(temp_tts_path)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("TTS returned an empty or invalid audio array")
            peak = float(np.max(np.abs(audio)))
            if peak < 1e-5:
                raise RuntimeError(
                    f"TTS returned effectively silent audio (peak={peak:.2e}); "
                    "the segment will not be silently omitted"
                )
            audio, sr = self._resample_to_target(audio, sr)
            
            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            final_overflow = len(audio) / sr - effective_duration

            # PASS 3: Truncate with fade-out if still overflowing
            if final_overflow > TOLERANCE_TOO_LONG and effective_duration > MIN_SEGMENT_DURATION:
                max_samples = int(effective_duration * sr)
                if max_samples < len(audio):
                    FADE_OUT_MS = 80  # 80ms smooth fade-out
                    fade_samples = min(int(FADE_OUT_MS / 1000 * sr), max_samples // 2)
                    audio = audio[:max_samples]
                    fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                    audio[-fade_samples:] *= fade
                    was_truncated = True
                    print(f"  Truncated [{segment['start']:.1f}-{segment['end']:.1f}]: {len(audio)/sr:.2f}s (fade-out {FADE_OUT_MS}ms)")

            cache_temporary = final_synced_path + ".writing.wav"
            metadata_path = final_synced_path + ".meta.json"
            metadata_temporary = metadata_path + ".tmp"
            sf.write(cache_temporary, audio, sr)
            # Verify the completed temporary WAV before exposing it as a reusable cache.
            cache_info = sf.info(cache_temporary)
            if cache_info.frames <= 0 or cache_info.samplerate <= 0:
                raise RuntimeError("temporary TTS cache failed integrity validation")
            os.replace(cache_temporary, final_synced_path)
            actual_duration = len(audio) / sr
            overflow_secs = max(0, actual_duration - effective_duration)
            with open(metadata_temporary, "w", encoding="utf-8") as handle:
                json.dump({"sped_up": was_sped_up, "speed_factor": speed_factor, "truncated": was_truncated,
                           "reformulated": final_text != text, "final_text": final_text}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(metadata_temporary, metadata_path)

            return {
                "synced_path": final_synced_path,
                "start": segment["start"],
                "end": segment["end"],
                "final_text": final_text,
                "sped_up": was_sped_up,
                "speed_factor": round(speed_factor, 4),
                "truncated": was_truncated,
                "reformulated": final_text != text,
                "final_duration": round(actual_duration, 3),
                "slot_duration": round(effective_duration, 3),
                "overflow": round(overflow_secs, 3),
            }
        except Exception as seg_err:
            message = (
                f"TTS failed for segment [{segment['start']:.1f}-{segment['end']:.1f}] "
                f"({lang_tag}): {seg_err}"
            )
            print(f"[ERROR] {message}")
            # Do not turn a failed spoken sentence into a silent placeholder:
            # fail the chunk so the user is not handed a misleadingly complete video.
            raise RuntimeError(message) from seg_err

    def sync_all(self, segments, language, voice_mapping=None, total_duration=None, gender="Woman"):
        """Sync all segments with global gap awareness.
        
        Returns:
            synced_segments: list of synced segment dicts
            stats: dict with sped_up, truncated, overflow counts and details
        """
        self._compute_effective_durations(segments, total_duration=total_duration)
        
        synced_segments = []
        sped_up_count = 0
        truncated_count = 0
        overflow_count = 0  # Audio overflows slot (crossfade handles)
        reformulated_count = 0
        sped_up_list = []
        truncated_list = []
        overflow_list = []
        
        total_segs = len(segments)
        for i, seg in enumerate(segments):
            voice = None
            if voice_mapping:
                spk = seg.get("speaker", "SPEAKER_00")
                voice = voice_mapping.get(spk)

            print(f"[TTS] Segment {i+1}/{total_segs} [{seg['start']:.1f}s-{seg['end']:.1f}s] generating...", flush=True)
            res = self.sync_segment(seg, language, voice_path=voice, gender=gender)
            print(f"[TTS] Segment {i+1}/{total_segs} done ({res.get('final_duration', res.get('duration', 0)):.2f}s)", flush=True)
            synced_segments.append(res)
            
            if res.get("sped_up"):
                sped_up_count += 1
                sped_up_list.append(f"[{seg['start']:.1f}-{seg['end']:.1f}]")
            if res.get("truncated"):
                truncated_count += 1
                truncated_list.append(f"[{seg['start']:.1f}-{seg['end']:.1f}]")
            elif res.get("overflow", 0) > 0.05:  # >50ms overflow (not truncated but will crossfade)
                overflow_count += 1
                overflow_list.append(f"[{seg['start']:.1f}-{seg['end']:.1f}] +{res['overflow']:.2f}s")
            if res.get("reformulated"):
                reformulated_count += 1
        
        cut_total = truncated_count + overflow_count
        stats = {
            "total": len(segments),
            "sped_up": sped_up_count,
            "truncated": truncated_count,
            "overflow": overflow_count,
            "reformulated": reformulated_count,
            "cut_total": cut_total,
            "perfect": len(segments) - sped_up_count - cut_total,
            "sped_up_segments": sped_up_list,
            "truncated_segments": truncated_list,
            "overflow_segments": overflow_list,
        }
        
        print(f"\n[Sync Stats] {stats['perfect']}/{stats['total']} perfect, "
              f"{sped_up_count} sped up, {truncated_count} truncated, {overflow_count} overflow")
        if truncated_list:
            print(f"  Truncated (cut with fade-out): {', '.join(truncated_list)}")
        if overflow_list:
            print(f"  Overflow (tail lost in crossfade): {', '.join(overflow_list)}")
        
        return synced_segments, stats

    # ---- NEVER CUT VOCAL MODE ----

    def _generate_all_natural(self, segments, language, voice_mapping=None, gender="Woman"):
        """
        Phase 1: Generate ALL TTS at natural speed (no speedup, no truncation).
        Returns segments enriched with tts_path and tts_duration (real measured).
        """
        results = []
        for i, seg in enumerate(segments):
            text = seg.get("translated_text", seg.get("text", ""))
            strict_duration = seg["end"] - seg["start"]

            if strict_duration < MIN_SEGMENT_DURATION or not text.strip():
                results.append({
                    **seg,
                    "tts_path": None,
                    "tts_duration": 0.0
                })
                continue

            lang_tag = str(language).lower().strip()[:3] if language else "all"
            voice = None
            if voice_mapping:
                spk = seg.get("speaker", "SPEAKER_00")
                voice = voice_mapping.get(spk)

            cache_tag = self._audio_cache_tag(
                text, language, seg["start"], seg["end"], voice, gender, 1.0,
            )
            cache_stem = f"nc_seg_{lang_tag}_{seg['start']:.2f}_{cache_tag}"
            synced_path = os.path.join(TEMP_DIR, f"{cache_stem}_synced.wav")
            tts_path = os.path.join(TEMP_DIR, f"{cache_stem}.wav")

            # Reuse only a readable, non-silent cache for this exact utterance/voice.
            if os.path.exists(synced_path):
                try:
                    info = sf.info(synced_path)
                    cached_audio, _cached_sr = sf.read(synced_path, dtype="float32", always_2d=False)
                    cache_valid = (info.duration > 0.05 and cached_audio.size > 0
                                   and np.isfinite(cached_audio).all()
                                   and float(np.max(np.abs(cached_audio))) > 1e-7)
                    del cached_audio
                    if cache_valid:
                        results.append({
                            **seg,
                            "tts_path": synced_path,
                            "tts_duration": info.duration
                        })
                        print(f"[NeverCut] Seg [{seg['start']:.1f}-{seg['end']:.1f}] ({lang_tag}): Reusing validated cache ({info.duration:.2f}s)")
                        continue
                    os.remove(synced_path)
                except Exception as e:
                    try:
                        os.remove(synced_path)
                    except OSError:
                        pass
                    print(f"[NeverCut] Invalid cached audio ({e}); regenerating this segment.")

            try:
                res = self.tts.generate(
                    text=text, language=language, output_path=tts_path, ref_audio_path=voice, speed=1.0,
                    duration=strict_duration, gender=gender
                )
                info = sf.info(tts_path)
                generated_audio, _generated_sr = sf.read(tts_path, dtype="float32", always_2d=False)
                if (info.duration <= 0.05 or generated_audio.size == 0
                        or not np.isfinite(generated_audio).all()
                        or float(np.max(np.abs(generated_audio))) <= 1e-7):
                    raise RuntimeError("TTS returned an empty or silent audio file.")
                gen_dur = float(info.duration)
                del generated_audio
            except Exception as e:
                try:
                    os.remove(tts_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Failed to synthesize never-cut segment [{seg['start']:.1f}-{seg['end']:.1f}] "
                    f"({lang_tag}); refusing to replace failed speech with silence: {e}"
                ) from e

            results.append({
                **seg,
                "tts_path": tts_path,
                "tts_duration": gen_dur
            })
            print(f"[NeverCut] Seg [{seg['start']:.1f}-{seg['end']:.1f}]: "
                  f"TTS={gen_dur:.2f}s (slot={strict_duration:.2f}s)")

        return results

    def _plan_cascade_placement(self, segments, total_duration):
        """
        Phase 2: Global gap analysis + cascade placement with pre-absorption.
        
        Pass 1: Naive cascade — place each segment at max(cursor, original_start).
        Pass 2 (if overflow): Pull segments earlier by consuming available gaps,
                so the audio fits within total_duration.
        
        Returns:
            segments: enriched with real_start / real_end
            drift_info: dict with max_drift, final_overflow, absorbed_by_gaps
        """
        # ---- PASS 1: Naive cascade to measure overflow ----
        cursor = 0.0
        gap_info = []  # [(segment_index, available_gap)]

        for i, seg in enumerate(segments):
            original_start = seg["start"]
            tts_dur = seg.get("tts_duration", 0.0)

            if tts_dur <= 0:
                seg["real_start"] = original_start
                seg["real_end"] = seg["end"]
                seg["drift"] = 0.0
                continue

            if cursor < original_start:
                available_gap = original_start - cursor
                gap_info.append((i, available_gap))
                real_start = original_start
            else:
                real_start = cursor

            seg["real_start"] = real_start
            seg["real_end"] = real_start + tts_dur
            seg["drift"] = real_start - original_start
            cursor = seg["real_end"]

        naive_overflow = max(0.0, cursor - total_duration)

        # ---- PASS 2: Pre-absorb overflow by consuming gaps ----
        if naive_overflow > 0.05 and gap_info:
            total_available = sum(g for _, g in gap_info)
            # How much gap to consume (up to the overflow amount)
            to_absorb = min(naive_overflow, total_available)
            
            if to_absorb > 0.01:
                print(f"[NeverCut] Overflow={naive_overflow:.2f}s — "
                      f"pre-absorbing {to_absorb:.2f}s from {len(gap_info)} gaps")
                
                # Distribute absorption proportionally across gaps
                absorbed_per_gap = {}
                for idx, gap_size in gap_info:
                    proportion = gap_size / total_available
                    absorbed_per_gap[idx] = proportion * to_absorb
                
                # Re-run placement with reduced gaps
                cursor = 0.0
                for i, seg in enumerate(segments):
                    original_start = seg["start"]
                    tts_dur = seg.get("tts_duration", 0.0)

                    if tts_dur <= 0:
                        seg["real_start"] = original_start
                        seg["real_end"] = seg["end"]
                        seg["drift"] = 0.0
                        continue

                    if cursor < original_start:
                        # Reduce gap by absorbed amount (start earlier)
                        gap_reduction = absorbed_per_gap.get(i, 0)
                        earliest_start = max(cursor, original_start - gap_reduction)
                        real_start = earliest_start
                    else:
                        real_start = cursor

                    seg["real_start"] = real_start
                    seg["real_end"] = real_start + tts_dur
                    seg["drift"] = real_start - original_start
                    cursor = seg["real_end"]

        # ---- Compute final stats ----
        max_drift = 0.0
        total_gap_absorbed = 0.0
        
        for seg in segments:
            if seg.get("tts_duration", 0) <= 0:
                continue
            drift = seg.get("drift", 0)
            max_drift = max(max_drift, drift)
            # Negative drift means we started early (pre-absorbed)
            early_start = max(0, seg["start"] - seg.get("real_start", seg["start"]))
            total_gap_absorbed += early_start
            
            if abs(drift) > 0.1:
                direction = "early" if drift < 0 else "late"
                print(f"[NeverCut] Seg [{seg['start']:.1f}-{seg['end']:.1f}]: "
                      f"placed at {seg['real_start']:.2f}s ({direction}: {abs(drift):.2f}s)")

        final_overflow = max(0.0, cursor - total_duration)

        drift_info = {
            "max_drift": round(max_drift, 2),
            "final_overflow": round(final_overflow, 2),
            "absorbed_by_gaps": round(total_gap_absorbed, 2),
            "pre_absorbed": round(naive_overflow - final_overflow, 2),
        }

        print(f"[NeverCut] Summary: max_drift={max_drift:.2f}s, "
              f"final_overflow={final_overflow:.2f}s, "
              f"pre-absorbed={drift_info['pre_absorbed']:.2f}s")

        return segments, drift_info

    def sync_all_never_cut(self, segments, language, total_duration, voice_mapping=None, gender="Woman"):
        """
        Never Cut Vocal mode: generate all TTS at natural speed, then
        plan cascade placement using global gap analysis.
        
        Returns:
            synced_segments: list of dicts with synced_path, real_start, real_end
            drift_info: dict with max_drift, final_overflow, absorbed_by_gaps
        """
        print(f"\n{'='*60}")
        print(f"NEVER CUT VOCAL MODE — {len(segments)} segments")
        print(f"{'='*60}")

        # Phase 1: Generate all TTS at natural speed
        enriched = self._generate_all_natural(segments, language, voice_mapping, gender=gender)

        # Phase 2: Plan cascade placement
        placed, drift_info = self._plan_cascade_placement(enriched, total_duration)

        # Phase 3: Build synced_segments list (same format as sync_all output)
        synced_segments = []
        lang_tag = str(language).lower().strip()[:3] if language else "all"
        for seg in placed:
            tts_path = seg.get("tts_path")
            synced_path = os.path.join(TEMP_DIR, f"nc_seg_{lang_tag}_{seg['start']:.2f}_synced.wav")

            if tts_path == synced_path and os.path.exists(synced_path):
                # Already resampled and cached
                pass
            elif tts_path and os.path.exists(tts_path):
                # Resample to target SR
                audio, sr = sf.read(tts_path)
                audio, sr = self._resample_to_target(audio, sr)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                sf.write(synced_path, audio, sr)
            else:
                synced_path = None

            synced_segments.append({
                "synced_path": synced_path,
                "start": seg["start"],
                "end": seg["end"],
                "real_start": seg.get("real_start", seg["start"]),
                "real_end": seg.get("real_end", seg["end"]),
                "final_text": seg.get("translated_text", seg.get("text", "")),
                "drift": seg.get("drift", 0.0)
            })

        print(f"{'='*60}\n")
        return synced_segments, drift_info

    def build_full_audio(self, synced_segments, total_duration, sr=OUTPUT_SAMPLE_RATE, use_real_positions=False, output_path=None):
        """Assemble the full voice track with crossfade at overlaps.
        
        If use_real_positions=True (Never Cut mode), uses seg['real_start'] for placement
        instead of seg['start'], allowing cascade-shifted positioning.
        """
        CROSSFADE_MS = 50  # 50ms crossfade at overlap boundaries
        crossfade_samples = int(CROSSFADE_MS / 1000 * sr)
        
        output_path = os.fspath(output_path) if output_path is not None else os.path.join(TEMP_DIR, "full_voice_track.wav")
        total_samples = int(total_duration * sr)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        block_frames = 65536

        # Preallocate silence on disk as FLOAT so crossfade math matches the
        # old float32 accumulator without holding a full-track array in RAM.
        with sf.SoundFile(output_path, mode="w", samplerate=sr, channels=1, subtype="FLOAT") as writer:
            remaining = total_samples
            silence = np.zeros(min(block_frames, max(1, total_samples)), dtype=np.float32)
            while remaining > 0:
                count = min(remaining, len(silence))
                writer.write(silence[:count])
                remaining -= count

        prev_audio_end = 0
        with sf.SoundFile(output_path, mode="r+") as output:
            for seg in synced_segments:
                source_path = seg.get("synced_path")
                if not source_path or not os.path.isfile(source_path):
                    continue

                source_path = os.fspath(source_path)
                info = sf.info(source_path)
                if info.samplerate != sr or info.channels != 1:
                    resampled_path = f"{source_path}.mono_{sr}.wav"
                    if not os.path.isfile(resampled_path) or sf.info(resampled_path).samplerate != sr:
                        import subprocess
                        subprocess.run(
                            ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", source_path,
                             "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", resampled_path],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        )
                    source_path = resampled_path
                    info = sf.info(source_path)

                placement_time = seg.get("real_start", seg["start"]) if use_real_positions else seg["start"]
                start_sample = max(0, min(total_samples, int(float(placement_time) * sr)))
                end_sample = min(start_sample + int(info.frames), total_samples)
                audio_len = end_sample - start_sample
                if audio_len <= 0:
                    continue

                overlap = prev_audio_end - start_sample
                xfade_len = min(crossfade_samples, audio_len, overlap) if overlap > crossfade_samples else 0

                with sf.SoundFile(source_path, mode="r") as source:
                    if xfade_len:
                        head = source.read(xfade_len, dtype="float32", always_2d=True).mean(axis=1)
                        output.flush()
                        output.seek(start_sample)
                        old = output.read(xfade_len, dtype="float32", always_2d=True)[:, 0]
                        fade_out = np.linspace(1.0, 0.0, xfade_len, dtype=np.float32)
                        fade_in = np.linspace(0.0, 1.0, xfade_len, dtype=np.float32)
                        output.seek(start_sample)
                        output.write(old * fade_out + head * fade_in)
                        source.seek(xfade_len)
                        destination = start_sample + xfade_len
                        remaining = audio_len - xfade_len
                    else:
                        destination = start_sample
                        remaining = audio_len
                        source.seek(0)

                    output.seek(destination)
                    while remaining > 0:
                        block = source.read(min(block_frames, remaining), dtype="float32", always_2d=True)
                        if not len(block):
                            break
                        mono = block.mean(axis=1)
                        output.write(mono)
                        remaining -= len(mono)

                prev_audio_end = max(prev_audio_end, end_sample)
        return output_path

if __name__ == "__main__":
    pass
