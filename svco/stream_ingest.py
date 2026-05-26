"""Audio capture, VAD-based routing, and turn finalization for sequential speech/chords."""

from __future__ import annotations

import time
from array import array
import audioop
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import pyaudio
import numpy as np
import torch
import time

SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SIZE = 512
SPEECH_THRESHOLD = 0.5
CHORD_RMS_THRESHOLD = 0.02
QUIET_SECONDS = 2.5
LOGGER = logging.getLogger(__name__)


@dataclass
class TurnRouter:
    """State router for splitting chunks into speech/chord buffers."""

    speech_threshold: float = SPEECH_THRESHOLD
    chord_rms_threshold: float = CHORD_RMS_THRESHOLD
    quiet_seconds: float = QUIET_SECONDS
    speech_buffer: bytearray = field(default_factory=bytearray)
    chord_buffer: bytearray = field(default_factory=bytearray)
    last_activity: float = field(default_factory=time.monotonic)

    @staticmethod
    def rms(audio_chunk: bytes) -> float:
        if not audio_chunk:
            return 0.0
        return float(audioop.rms(audio_chunk, 2) / 32768.0)

    def route_chunk(self, audio_chunk: bytes, speech_probability: float, now: Optional[float] = None) -> None:
        current_time = time.monotonic() if now is None else now
        chunk_rms = self.rms(audio_chunk)

        if speech_probability > self.speech_threshold:
            self.speech_buffer.extend(audio_chunk)
            self.last_activity = current_time
            return

        if chunk_rms > self.chord_rms_threshold:
            self.chord_buffer.extend(audio_chunk)
            self.last_activity = current_time

    def should_flush(self, now: Optional[float] = None) -> bool:
        current_time = time.monotonic() if now is None else now
        has_content = bool(self.speech_buffer or self.chord_buffer)
        return has_content and (current_time - self.last_activity) >= self.quiet_seconds

    def flush(self) -> Tuple[bytes, bytes]:
        speech_bytes = bytes(self.speech_buffer)
        chord_bytes = bytes(self.chord_buffer)
        self.speech_buffer.clear()
        self.chord_buffer.clear()
        self.last_activity = time.monotonic()
        LOGGER.info("Turn finished")
        return speech_bytes, chord_bytes
        
def run_stream_ingest():
    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, *_) = utils

    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']} (max input channels: {info['maxInputChannels']})")

    device_index = int(input("Enter device index to use for microphone: ").strip())

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index
    )

    print("* Recording... (speak, play chords, stay silent 2.5s to trigger flush, Ctrl+C to quit)")

    speech_buffer = bytearray()
    chord_buffer = bytearray()
    last_active_time = time.time()

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0 # normalize

            audio_tensor = torch.from_numpy(audio_array)
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            speech_prob = model(audio_tensor, RATE).item()
            rms = np.sqrt(np.mean(audio_array**2))
            t_now = time.time()

            print(f"VAD speech prob: {speech_prob:.3f} | RMS: {rms:.2f}")

            active = False
            # Route
            if speech_prob > 0.5:
                speech_buffer.extend(data)
                print("[router] Chunk → SPEECH buffer.")
                active = True
            elif rms > 0.01:  # Change threshold as needed for your environment
                chord_buffer.extend(data)
                print("[router] Chunk → CHORD buffer.")
                active = True
            # else: silence

            if active:
                last_active_time = t_now

            # If silence for >2.5s and have data: FLUSH buffers ("turn finished")
            if (t_now - last_active_time) > 2.5 and (speech_buffer or chord_buffer):
                print("\nTurn Finished! Flushing buffers to next stage.")
                print(f"  Speech buffer: {len(speech_buffer)} bytes | Chord buffer: {len(chord_buffer)} bytes\n")
                # ----->>> Handoff to next pipeline step (return)
                # We'll return and let main.py continue the pipeline
                return speech_buffer, chord_buffer

    except KeyboardInterrupt:
        print("* Done recording")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def _load_silero_vad_model():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Silero VAD.") from exc

    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model


def _speech_probability(vad_model, audio_chunk: bytes) -> float:
    import torch

    if not audio_chunk:
        return 0.0

    pcm = array("h")
    pcm.frombytes(audio_chunk)
    if len(pcm) == 0:
        return 0.0

    audio_tensor = torch.tensor([sample / 32768.0 for sample in pcm], dtype=torch.float32)
    with torch.no_grad():
        probability = vad_model(audio_tensor, SAMPLE_RATE)
    return float(probability.item())


def stream_turns(
    flush_callback: Callable[[bytes, bytes], None],
    *,
    vad_model=None,
    router: Optional[TurnRouter] = None,
) -> None:
    """Run an infinite loop reading microphone chunks and flushing completed turns."""
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError("PyAudio is required for stream ingestion.") from exc

    model = vad_model or _load_silero_vad_model()
    turn_router = router or TurnRouter()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    try:
        while True:
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            speech_prob = _speech_probability(model, audio_chunk)
            turn_router.route_chunk(audio_chunk, speech_prob)

            if turn_router.should_flush():
                speech_bytes, chord_bytes = turn_router.flush()
                flush_callback(speech_bytes, chord_bytes)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
