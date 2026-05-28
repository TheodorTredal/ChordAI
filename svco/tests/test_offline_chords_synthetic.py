import math
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from svco.offline_chords import transcribe_file


def _sine(sr: int, freq: float, dur_s: float) -> np.ndarray:
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    return np.sin(2 * math.pi * freq * t).astype(np.float32)


def _chord(sr: int, freqs: list[float], dur_s: float) -> np.ndarray:
    y = sum(_sine(sr, f, dur_s) for f in freqs)
    y /= max(np.max(np.abs(y)), 1e-6)
    # Add a tiny attack/decay envelope
    n = y.shape[0]
    env = np.ones((n,), dtype=np.float32)
    a = int(0.01 * sr)
    if a > 1:
        env[:a] = np.linspace(0, 1, a)
        env[-a:] = np.linspace(1, 0.2, a)
    return y * env


def test_synthetic_progression_detects_expected_labels():
    # Build a simple progression: Am | F | G7 | Dsus4
    sr = 16000
    dur = 2.0

    # Frequencies (A4=440) basic equal temperament
    def hz(note: str) -> float:
        # minimal mapping for test notes
        mapping = {
            "A3": 220.0,
            "C4": 261.6256,
            "E4": 329.6276,
            "F3": 174.6141,
            "A3_": 220.0,
            "C4_": 261.6256,
            "G3": 196.0,
            "B3": 246.9417,
            "D4": 293.6648,
            "F4": 349.2282,
            "D3": 146.8324,
        }
        return mapping[note]

    am = _chord(sr, [hz("A3"), hz("C4"), hz("E4")], dur)
    f = _chord(sr, [hz("F3"), hz("A3_"), hz("C4_")], dur)
    g7 = _chord(sr, [hz("G3"), hz("B3"), hz("D4"), hz("F4")], dur)
    dsus4 = _chord(sr, [hz("D3"), hz("G3"), hz("A3")], dur)

    y = np.concatenate([am, f, g7, dsus4])

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prog.wav"
        sf.write(p, y, sr)

        res = transcribe_file(str(p), target_sr=22050, hop_s=0.10, min_seg_s=0.5, change_cost=2.0)
        segs = [s for s in res["segments"] if s["label"] != "N"]
        labels = [s["display"] for s in segs]

        # We allow some boundary noise; just check the expected labels appear in order.
        expected = ["Am", "F", "G7", "Dsus4"]
        idx = 0
        for lab in labels:
            if idx < len(expected) and lab.startswith(expected[idx]):
                idx += 1
        assert idx == len(expected), f"Expected sequence {expected}, got {labels}"
