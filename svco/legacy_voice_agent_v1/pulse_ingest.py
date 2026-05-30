from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
CHORD_START_RMS_THRESHOLD = 0.006   # start chord capture
CHORD_KEEP_RMS_THRESHOLD = 0.004    # keep capturing once started (lower than start)
CHORD_HANGOVER_S = 0.60            # keep capturing for 400ms after last "strong" frame
MIN_CHORD_TURN_S = 1.20            # don't flush chord-only turns before ~0.8s collected
# Use 16k for Silero VAD and for your transcription_engine SAMPLE_RATE
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # s16le
CHUNK_SAMPLES = 512
CHUNK_BYTES = CHUNK_SAMPLES * CHANNELS * SAMPLE_WIDTH_BYTES

QUIET_SECONDS = 1.2
SPEECH_THRESHOLD = 0.85

# NOTE: your observed "playing" RMS looked like ~0.012–0.016, and silence ~0.000–0.001.
# So 0.01 is a reasonable starting point; adjust up/down based on route debug.
CHORD_RMS_THRESHOLD = 0.005

# 4 frames is ~128ms at 16kHz/512; can be too strict if RMS hovers around the threshold.
# 2 is more responsive; increase if you get false positives.
CHORD_STREAK_FRAMES = 2


@dataclass
class TurnRouter:
    speech_buffer: bytearray = field(default_factory=bytearray)
    chord_buffer: bytearray = field(default_factory=bytearray)
    chord_mode: bool = False
    chord_hangover_until: float = 0.0
    last_activity: float = field(default_factory=time.monotonic)
    turn_start: float = field(default_factory=time.monotonic)
    rms_ema: float = 0.0

    chord_active_streak: int = 0

    # --- debug ---
    debug: bool = True
    _last_flush_debug_print: float = field(default=0.0, init=False, repr=False)

    @staticmethod
    def rms_from_int16(pcm16: np.ndarray) -> float:
        if pcm16.size == 0:
            return 0.0
        x = pcm16.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x)))

    def _dbg_flush_state(self, t: float, *, reason: str, max_turn_seconds: float) -> None:
        """Print state relevant to flushing; throttled to ~1Hz."""
        if not self.debug:
            return
        if (t - self._last_flush_debug_print) < 1.0:
            return
        self._last_flush_debug_print = t

        dt_quiet = t - self.last_activity
        dt_turn = t - self.turn_start
        print(
            "[flush-check]"
            f" reason={reason}"
            f" has_speech={len(self.speech_buffer)}"
            f" has_chord={len(self.chord_buffer)}"
            f" dt_quiet={dt_quiet:.3f}s"
            f" dt_turn={dt_turn:.3f}s"
            f" quiet_seconds={QUIET_SECONDS}"
            f" max_turn={max_turn_seconds}"
        )

    def route(self, chunk_bytes: bytes, speech_probability: float, now: Optional[float] = None) -> None:
        t = time.monotonic() if now is None else now

        pcm16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        rms = self.rms_from_int16(pcm16)
        alpha = 0.15  # higher = more responsive
        self.rms_ema = (1 - alpha) * self.rms_ema + alpha * rms
        rms_s = self.rms_ema
        def start_turn_if_needed() -> None:
            if not self.speech_buffer and not self.chord_buffer:
                self.turn_start = t

        # --- SPEECH wins ---
        if speech_probability > SPEECH_THRESHOLD:
            start_turn_if_needed()
            self.speech_buffer.extend(chunk_bytes)
            self.last_activity = t

            if not self.chord_mode:
                self.chord_active_streak = 0
                self.chord_hangover_until = 0.0

            if self.debug:
                print(f"[route] SPEECH vad={speech_probability:.3f} rms={rms:.3f} speech_bytes={len(self.speech_buffer)}")
    

        # --- Update chord start streak ---
        if rms_s >= CHORD_START_RMS_THRESHOLD:
            self.chord_active_streak += 1
        else:
            # Be slightly more forgiving before resetting the streak to 0
            if self.chord_active_streak > 0 and rms_s < CHORD_KEEP_RMS_THRESHOLD:
               self.chord_active_streak = 0
        # --- Enter chord mode once we see N consecutive "strong" frames ---
        if (not self.chord_mode) and (self.chord_active_streak >= CHORD_STREAK_FRAMES):
            self.chord_mode = True
            if self.debug:
                print(f"[route] CHORD_START rms={rms_s:.3f} streak={self.chord_active_streak}/{CHORD_STREAK_FRAMES}")

        # --- If in chord mode, buffer with hangover ---
        if self.chord_mode:
            start_turn_if_needed()
            self.chord_buffer.extend(chunk_bytes)
            self.last_activity = t

            # refresh hangover when we have at least "keep" energy
            if rms_s >= CHORD_KEEP_RMS_THRESHOLD:
                self.chord_hangover_until = t + CHORD_HANGOVER_S

            if self.debug:
                print(
                    f"[route] CHORD rms={rms_s:.3f} chord_bytes={len(self.chord_buffer)} "
                    f"hangover_left={max(0.0, self.chord_hangover_until - t):.3f}s"
                )

            # exit chord mode only after hangover expires
            if t >= self.chord_hangover_until:
                if self.debug:
                    print(f"[route] CHORD_END rms={rms_s:.3f} chord_bytes={len(self.chord_buffer)}")
                self.chord_mode = False
                self.chord_active_streak = 0
            return

    def should_flush(self, now: Optional[float] = None, *, max_turn_seconds: float = 10.0) -> bool:
        t = time.monotonic() if now is None else now
        has_content = bool(self.speech_buffer or self.chord_buffer)

        if not has_content:
            self._dbg_flush_state(t, reason="no-content", max_turn_seconds=max_turn_seconds)
            return False

        quiet_done = (t - self.last_activity) >= QUIET_SECONDS
        max_done = (t - self.turn_start) >= max_turn_seconds

        # NEW: enforce minimum chord duration if this is a chord-only turn
        if self.chord_buffer and not self.speech_buffer:
            chord_s = _seconds_from_bytes(len(self.chord_buffer))
            if chord_s < MIN_CHORD_TURN_S and quiet_done and not max_done:
                self._dbg_flush_state(t, reason=f"waiting(min_chord {chord_s:.2f}s<{MIN_CHORD_TURN_S}s)", max_turn_seconds=max_turn_seconds)
                return False

        if quiet_done:
            self._dbg_flush_state(t, reason="quiet_done=True", max_turn_seconds=max_turn_seconds)
        elif max_done:
            self._dbg_flush_state(t, reason="max_done=True", max_turn_seconds=max_turn_seconds)
        else:
            self._dbg_flush_state(t, reason="waiting", max_turn_seconds=max_turn_seconds)

        return quiet_done or max_done
    def flush(self) -> Tuple[bytes, bytes]:
        s = bytes(self.speech_buffer)
        c = bytes(self.chord_buffer)

        self.speech_buffer.clear()
        self.chord_buffer.clear()

        now = time.monotonic()
        self.last_activity = now
        self.turn_start = now
        self.chord_active_streak = 0

        if self.debug:
            print(f"[flush] speech_bytes={len(s)} chord_bytes={len(c)}")

        return s, c

def run_pulse_ingest_continuous(short_pause_s=1.2, long_pause_s=4.0, device="RDPSource"):
    """
    Yields events continuously. 
    - ("sub_turn", (speech_bytes, chord_bytes)) on a short pause.
    - ("session_complete", None) on a long pause.
    """
    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )

    proc = _start_parec(device=device)
    if proc.stdout is None:
        raise RuntimeError("parec stdout pipe not available")

    router = TurnRouter()
    buf = bytearray()
    session_active = False

    print(f"* Continuous capture started. Short pause: {short_pause_s}s. Long pause: {long_pause_s}s. Ctrl+C to quit.")

    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break

            if len(chunk) != CHUNK_BYTES:
                buf.extend(chunk)
                while len(buf) >= CHUNK_BYTES:
                    frame = bytes(buf[:CHUNK_BYTES])
                    del buf[:CHUNK_BYTES]
                    now = time.monotonic()
                    _process_frame(model, router, frame, now=now)
            else:
                now = time.monotonic()
                _process_frame(model, router, chunk, now=now)

            # Evaluate timers
            has_content = bool(router.speech_buffer or router.chord_buffer)
            quiet_time = now - router.last_activity

            if has_content:
                session_active = True

            # 1. Short Pause -> Yield audio for background processing
            if has_content and quiet_time >= short_pause_s:
                yield "sub_turn", router.flush()

            # 2. Long Pause -> Yield completion signal to trigger Claude
            if session_active and not has_content and quiet_time >= long_pause_s:
                yield "session_complete", None
                session_active = False

    finally:
        try:
            proc.terminate()
        except Exception:
            pass




def _start_parec(device: str = "RDPSource") -> subprocess.Popen:
    """
    Start parec producing raw PCM:
      - s16le
      - mono
      - 16 kHz
    """
    cmd = [
        "parec",
        f"--device={device}",
        "--format=s16le",
        f"--channels={CHANNELS}",
        f"--rate={SAMPLE_RATE}",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _seconds_from_bytes(n_bytes: int) -> float:
    return n_bytes / float(SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS)

def run_pulse_turn_ingest(*, device: str = "RDPSource") -> Tuple[bytes, bytes]:
    """
    Capture audio from WSLg PulseAudio via parec and split into speech/chord buffers.
    Returns (speech_bytes, chord_bytes) when silence timeout triggers.
    """
    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )

    proc = _start_parec(device=device)
    if proc.stdout is None:
        raise RuntimeError("parec stdout pipe not available")

    router = TurnRouter()
    buf = bytearray()

    print("* Recording via WSLg PulseAudio (parec). Stay silent ~2.5s to flush. Ctrl+C to quit.")

    try:
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                err = b""
                if proc.stderr is not None:
                    try:
                        err = proc.stderr.read() or b""
                    except Exception:
                        err = b""
                raise RuntimeError(f"parec produced no audio. stderr={err.decode('utf-8', 'ignore')[:300]}")

            # Handle short reads by re-buffering into exact frames
            if len(chunk) != CHUNK_BYTES:
                buf.extend(chunk)
                while len(buf) >= CHUNK_BYTES:
                    frame = bytes(buf[:CHUNK_BYTES])
                    del buf[:CHUNK_BYTES]

                    frame_now = time.monotonic()
                    _process_frame(model, router, frame, now=frame_now)

                    if router.should_flush(now=frame_now):
                        return router.flush()
                continue

            now = time.monotonic()
            _process_frame(model, router, chunk, now=now)

            if router.should_flush(now=now):
                return router.flush()

    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def _process_frame(model, router: TurnRouter, frame_bytes: bytes, *, now: float) -> None:
    pcm16 = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    x = torch.from_numpy(pcm16).unsqueeze(0)
    with torch.no_grad():
        speech_prob = float(model(x, SAMPLE_RATE).item())

    rms = router.rms_from_int16(np.frombuffer(frame_bytes, dtype=np.int16))
    print(f"VAD={speech_prob:.3f} RMS={rms:.3f}")

    router.route(frame_bytes, speech_prob, now=now)