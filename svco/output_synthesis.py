"""Dual-channel response output handling for UI and local speech synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypedDict


class DualChannelResponse(TypedDict):
    speech_text: str
    chord_data: List[str]


@dataclass
class LocalSpeechSynthesizer:
    voice: str = "af_sarah"
    _pipeline: object = field(default=None, init=False, repr=False)

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("kokoro package is required for local TTS.") from exc

        self._pipeline = KPipeline(voice=self.voice)
        return self._pipeline

    def speak_stream(self, speech_text: str) -> None:
        if not speech_text.strip():
            return

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice package is required for local playback.") from exc

        pipeline = self._ensure_pipeline()
        for chunk in pipeline(speech_text):
            audio = getattr(chunk, "audio", None)
            sample_rate = getattr(chunk, "sample_rate", 24_000)
            if audio is None:
                continue
            sd.play(audio, sample_rate)
            sd.wait()


def parse_response_json(raw_response: str) -> DualChannelResponse:
    data = json.loads(raw_response)
    speech_text = data.get("speech_text")
    chord_data = data.get("chord_data")

    if not isinstance(speech_text, str):
        raise ValueError("Response must contain `speech_text` as a string.")
    if not isinstance(chord_data, list) or not all(isinstance(chord, str) for chord in chord_data):
        raise ValueError("Response must contain `chord_data` as a list of strings.")

    return {"speech_text": speech_text, "chord_data": chord_data}


def push_chords_to_ui(chord_data: List[str], ui_chord_buffer: Optional[List[str]] = None) -> List[str]:
    target = ui_chord_buffer if ui_chord_buffer is not None else []
    target.clear()
    target.extend(chord_data)
    return target

def synthesize_output(response_json):
    # Stub: Just print results for now
    print("[output_synthesis] Displaying output:")
    print("Speech Text:", response_json.get("speech_text"))
    print("Chord Data:", response_json.get("chord_data"))

    
def process_agent_response(
    raw_response: str,
    ui_chord_buffer: Optional[List[str]] = None,
    synthesizer: Optional[LocalSpeechSynthesizer] = None,
) -> Tuple[str, List[str]]:
    parsed = parse_response_json(raw_response)
    chord_data = push_chords_to_ui(parsed["chord_data"], ui_chord_buffer)

    if synthesizer is not None:
        synthesizer.speak_stream(parsed["speech_text"])

    return parsed["speech_text"], chord_data
