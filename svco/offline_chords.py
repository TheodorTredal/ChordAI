"""Offline chord transcription for mic'd acoustic guitar.

This module implements a simple but practical offline chord recognizer:
- CQT chroma (librosa)
- template matching (maj/min/7/sus2/sus4/no3 + N)
- Viterbi decoding for temporal smoothing
- segmentation into timestamped chord regions

It is intended as an *offline middle layer* that produces a stable chord timeline
suitable for downstream summarization / LLM prompting.

CLI:
  python -m offline_chords path/to/audio.wav --out chords.json

Notes:
- This is not state-of-the-art ML chord recognition.
- It is a strong baseline compared to mean-chroma heuristics and works well
  enough to support prompt conditioning.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf


PITCH_CLASSES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_TO_I = {pc: i for i, pc in enumerate(PITCH_CLASSES_SHARP)}


# ----------------------------
# Label normalization
# ----------------------------

_ENHARMONIC_FLATS = {
    "DB": "C#",
    "EB": "D#",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}

_SOLFEGE_SHARP = {
    "CS": "C#",
    "DS": "D#",
    "FS": "F#",
    "GS": "G#",
    "AS": "A#",
}


def normalize_label(label: str) -> str:
    """Normalize incoming chord labels to a canonical spelling.

    Supports:
      - csmin -> C#min
      - c#min / C#min -> C#min
      - Dbmin -> C#min
      - Cno3 -> Cno3
      - N / NC / NOCHORD -> N

    Output style:
      - Root is one of: C C# D D# E F F# G G# A A# B
      - Quality is one of: maj, min, 7, sus2, sus4, no3
      - Or the special label: N
    """

    if not label:
        return "N"

    s = label.strip()
    if not s:
        return "N"

    u = s.upper().replace(" ", "")

    if u in {"N", "NC", "NOCHORD", "NO-CHORD", "NONE"}:
        return "N"

    # Keep powerchords explicit
    # Accept variants: CNO3, C(no3)
    if "NO3" in u:
        # Extract root at the start
        root_raw = u.split("NO3", 1)[0]
        root = _normalize_root_token(root_raw)
        return f"{root}no3" if root != "N" else "N"

    # Detect root token (e.g. CS, C#, DB, C)
    root, rest = _split_root_and_rest(u)
    if root == "N":
        return "N"

    # Quality parsing
    if rest in {"", "MAJ", "MAJOR"}:
        qual = "maj"
    elif rest in {"M", "MIN", "MINOR"}:
        qual = "min"
    elif rest in {"7", "DOM7"}:
        qual = "7"
    elif rest in {"SUS2"}:
        qual = "sus2"
    elif rest in {"SUS4", "SUS"}:
        qual = "sus4"
    else:
        # Handle compact forms like C#M, C#MIN, etc.
        # If unknown, fall back to N rather than hallucinating.
        return "N"

    return f"{root}{qual}" if qual not in {"maj", "min"} else f"{root}{qual}"


def _split_root_and_rest(u: str) -> Tuple[str, str]:
    # Root can be:
    # - single letter A-G
    # - letter + '#' (e.g. C#)
    # - solfege sharp token (CS)
    # - flat token (DB)

    if not u:
        return "N", ""

    # If it already contains '#'
    if len(u) >= 2 and u[1] == "#":
        root = u[:2]
        rest = u[2:]
        root = _normalize_root_token(root)
        return root, rest

    # solfege sharps like CS, DS
    if len(u) >= 2 and u[:2] in _SOLFEGE_SHARP:
        root = _SOLFEGE_SHARP[u[:2]]
        rest = u[2:]
        return root, rest

    # flats like DB
    if len(u) >= 2 and u[:2] in _ENHARMONIC_FLATS:
        root = _ENHARMONIC_FLATS[u[:2]]
        rest = u[2:]
        return root, rest

    # single letter root
    if u[0] in {"A", "B", "C", "D", "E", "F", "G"}:
        root = u[0]
        rest = u[1:]
        root = _normalize_root_token(root)
        return root, rest

    return "N", u


def _normalize_root_token(tok: str) -> str:
    t = tok.upper().replace(" ", "")

    if t in _SOLFEGE_SHARP:
        return _SOLFEGE_SHARP[t]

    if t in _ENHARMONIC_FLATS:
        return _ENHARMONIC_FLATS[t]

    if t in PC_TO_I:
        return t

    # Handle lowercase like c#
    if len(t) == 2 and t[0] in {"A", "B", "C", "D", "E", "F", "G"} and t[1] == "#":
        return t

    if len(t) == 1 and t in {"A", "B", "C", "D", "E", "F", "G"}:
        return t

    return "N"


def format_label(root: str, quality: str) -> str:
    if quality == "N" or root == "N":
        return "N"
    if quality == "maj":
        return root
    if quality == "min":
        return f"{root}m"  # display style: Am, C#m
    if quality in {"7", "sus2", "sus4", "no3"}:
        return f"{root}{quality}"
    return "N"


# ----------------------------
# Templates / vocab
# ----------------------------

@dataclass(frozen=True)
class ChordState:
    root_i: int  # 0..11
    quality: str  # maj|min|7|sus2|sus4|no3|N

    @property
    def label_key(self) -> str:
        if self.quality == "N":
            return "N"
        return f"{PITCH_CLASSES_SHARP[self.root_i]}{self.quality}"


def iter_states(include_nochord: bool = True) -> List[ChordState]:
    qualities = ["maj", "min", "7", "sus2", "sus4", "no3"]
    states = [ChordState(root_i=r, quality=q) for q in qualities for r in range(12)]
    if include_nochord:
        states.append(ChordState(root_i=0, quality="N"))
    return states


def build_templates(states: Sequence[ChordState]) -> np.ndarray:
    """Return template matrix T of shape (num_states, 12) with L2-normalized rows."""

    intervals = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "7": [0, 4, 7, 10],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "no3": [0, 7],
        "N": [],
    }

    T = np.zeros((len(states), 12), dtype=np.float32)
    for si, st in enumerate(states):
        if st.quality == "N":
            # No-chord template is all zeros; special-cased later.
            continue
        for k in intervals[st.quality]:
            T[si, (st.root_i + k) % 12] = 1.0

    # L2 normalize non-zero templates
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    T = T / norms
    return T


# ----------------------------
# Audio / features
# ----------------------------

def load_audio_mono(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    if sr != target_sr:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Prevent clipping / huge gain changes; just normalize a bit.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = y / max(peak, 1.0)

    return y.astype(np.float32), sr


def compute_features(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    bins_per_octave: int = 36,
    fmin_hz: float = 65.40639132514966,  # C2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (chroma, rms, bass_pc) where:
       - chroma is (12, frames)
       - rms is (frames,)
       - bass_pc is (12, frames) approximate low-frequency pitch-class energy
    """
    import librosa

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        fmin=fmin_hz,
    ).astype(np.float32)

    cqt = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin_hz,
            n_bins=6 * bins_per_octave,  # ~6 octaves
            bins_per_octave=bins_per_octave,
        )
    ).astype(np.float32)  # shape: (bins, frames)

    # Focus on lowest ~1 octave to approximate bass/root (sharper than 2 octaves)
    low_bins = min(cqt.shape[0], 1 * bins_per_octave)
    cqt_low = cqt[:low_bins, :]

    # Fold bins -> pitch class (12) using max-pooling to emphasize the strongest bass partial
    bass_pc = np.zeros((12, cqt_low.shape[1]), dtype=np.float32)
    for b in range(cqt_low.shape[0]):
        pc = b % 12
        bass_pc[pc, :] = np.maximum(bass_pc[pc, :], cqt_low[b, :])

    # Normalize per frame (keep it peaky)
    bass_denom = np.sum(bass_pc, axis=0, keepdims=True)
    bass_denom = np.where(bass_denom == 0, 1.0, bass_denom)
    bass_pc = bass_pc / bass_denom

    # L2 normalize each frame (avoid division by 0)
    denom = np.linalg.norm(chroma, axis=0, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    chroma = chroma / denom

    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=2048)[0].astype(np.float32)
    # Ensure same frame count
    n = min(chroma.shape[1], rms.shape[0])
    n2 = min(n, bass_pc.shape[1])
    return chroma[:, :n2], rms[:n2], bass_pc[:, :n2]


# ----------------------------
# Decoding
# ----------------------------

def score_frames(chroma: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Cosine similarity since both are normalized -> dot product.

    Returns scores of shape (num_states, frames).
    """

    return templates @ chroma  # (S,12) x (12,T) -> (S,T)


def _log_softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    y = x - m
    e = np.exp(y)
    s = np.sum(e, axis=axis, keepdims=True)
    return y - np.log(s + 1e-12)


def viterbi_decode(
    log_emissions: np.ndarray,
    change_cost: float = 2.0,
) -> np.ndarray:
    """Simple Viterbi with uniform transition costs.

    log_emissions: shape (S,T)
    change_cost: penalty (positive) applied when switching states.

    Returns best path as int array of shape (T,)
    """

    S, T = log_emissions.shape
    dp = np.full((S, T), -np.inf, dtype=np.float32)
    back = np.zeros((S, T), dtype=np.int32)

    # Start prior: uniform
    dp[:, 0] = log_emissions[:, 0]

    for t in range(1, T):
        prev = dp[:, t - 1]

        # Best previous state if we were to change from some other state.
        best_prev_state = int(np.argmax(prev))
        best_prev_score = float(prev[best_prev_state])

        for s in range(S):
            stay_score = float(prev[s])
            change_score = best_prev_score - (0.0 if best_prev_state == s else change_cost)
            if stay_score >= change_score:
                dp[s, t] = stay_score + log_emissions[s, t]
                back[s, t] = s
            else:
                dp[s, t] = change_score + log_emissions[s, t]
                back[s, t] = best_prev_state

    path = np.zeros((T,), dtype=np.int32)
    path[T - 1] = int(np.argmax(dp[:, T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[path[t + 1], t + 1]
    return path


def path_to_segments(
    path: np.ndarray,
    states: Sequence[ChordState],
    hop_length: int,
    sr: int,
    min_seg_s: float = 0.4,
    confidence: Optional[np.ndarray] = None,
) -> List[dict]:
    """Convert decoded state path into merged segments."""

    hop_s = hop_length / float(sr)

    segs: List[dict] = []
    if path.size == 0:
        return segs

    start = 0
    cur = int(path[0])

    def flush(end_idx: int):
        nonlocal start, cur
        st = states[cur]
        start_t = start * hop_s
        end_t = end_idx * hop_s
        dur = end_t - start_t
        if dur <= 0:
            return

        # Confidence: mean over frames in segment if provided, else None
        conf = None
        if confidence is not None:
            conf = float(np.mean(confidence[start:end_idx]))

        segs.append(
            {
                "start": round(start_t, 3),
                "end": round(end_t, 3),
                "label": st.label_key,
                "confidence": None if conf is None else round(conf, 3),
            }
        )

    for i in range(1, path.size):
        s = int(path[i])
        if s != cur:
            flush(i)
            start = i
            cur = s

    flush(path.size)

    # Merge short segments into neighbors
    merged: List[dict] = []
    for seg in segs:
        if not merged:
            merged.append(seg)
            continue

        if seg["end"] - seg["start"] < min_seg_s:
            # Merge into previous
            merged[-1]["end"] = seg["end"]
            # keep previous label; confidence: weighted average if available
            continue

        if seg["label"] == merged[-1]["label"]:
            merged[-1]["end"] = seg["end"]
            continue

        merged.append(seg)

    # Drop very short leftovers
    merged = [m for m in merged if (m["end"] - m["start"]) >= min_seg_s or m["label"] == "N"]
    return merged


def transcribe_file(
    wav_path: str,
    *,
    target_sr: int = 22050,
    hop_s: float = 0.10,
    bins_per_octave: int = 36,
    min_seg_s: float = 0.40,
    change_cost: float = 2.0,
    rms_threshold: float = 0.01,
) -> dict:
    """Offline chord transcription returning a timeline of segments."""

    y, sr = load_audio_mono(wav_path, target_sr=target_sr)
    hop_length = max(32, int(round(sr * hop_s)))

    chroma, rms, bass_pc = compute_features(
        y,
        sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        fmin_hz=65.40639132514966,  # C2
    )

    states = iter_states(include_nochord=True)
    templates = build_templates(states)

    scores = score_frames(chroma, templates)
    bass_weight = 1.25
    for si, st in enumerate(states):
        if st.quality == "N":
            continue
        scores[si, :] += bass_weight * bass_pc[st.root_i, :]

    # Add a no-chord score based on RMS gate: if quiet -> strong N
    # Last state is N
    n_state = len(states) - 1
    quiet = rms < rms_threshold
    scores[n_state, :] = np.where(quiet, 1.0, -0.5).astype(np.float32)

    # Convert to log probabilities per frame
    log_em = _log_softmax(scores, axis=0)

    # A simple confidence proxy: exp(margin) between best and second best
    # (computed pre-viterbi, used for segment avg)
    top2 = np.partition(scores, -2, axis=0)[-2:, :]
    margin = (top2[1, :] - top2[0, :]).astype(np.float32)  # note: may be negative due to ordering
    # Actually ensure best-second
    best = np.max(scores, axis=0)
    second = np.max(np.where(scores == best, -np.inf, scores), axis=0)
    margin = (best - second).astype(np.float32)
    frame_conf = 1.0 / (1.0 + np.exp(-margin))  # squashed

    path = viterbi_decode(log_em, change_cost=change_cost)

    segs = path_to_segments(
        path,
        states=states,
        hop_length=hop_length,
        sr=sr,
        min_seg_s=min_seg_s,
        confidence=frame_conf,
    )

    # Convert internal label_key (e.g. C#min) to display style (C#m etc)
    # Keep label_key too for evaluation/consistency.
    for seg in segs:
        key = seg["label"]
        if key == "N":
            seg["display"] = "N"
            continue
        # Parse key like "C#min" "G7" "Cno3" "Dsus4" "Fmaj"
        root, qual = _split_key_to_root_qual(key)
        seg["display"] = format_label(root, qual)

    return {
        "source": str(wav_path),
        "sr": sr,
        "hop_s": hop_s,
        "segments": segs,
        "params": {
            "target_sr": target_sr,
            "bins_per_octave": bins_per_octave,
            "min_seg_s": min_seg_s,
            "change_cost": change_cost,
            "rms_threshold": rms_threshold,
        },
    }


def _split_key_to_root_qual(key: str) -> Tuple[str, str]:
    if key == "N":
        return "N", "N"
    # Root can be like C#
    if len(key) >= 2 and key[1] == "#":
        root = key[:2]
        rest = key[2:]
    else:
        root = key[:1]
        rest = key[1:]

    # rest starts with quality: maj|min|7|sus2|sus4|no3
    if rest.startswith("maj"):
        return root, "maj"
    if rest.startswith("min"):
        return root, "min"
    if rest.startswith("sus2"):
        return root, "sus2"
    if rest.startswith("sus4"):
        return root, "sus4"
    if rest.startswith("no3"):
        return root, "no3"
    if rest.startswith("7"):
        return root, "7"
    # Fallback
    return root, "maj"


# ----------------------------
# CLI
# ----------------------------

def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Offline chord transcription (CQT + templates + Viterbi)")
    p.add_argument("wav", help="Path to WAV/AIFF/FLAC (anything soundfile can read)")
    p.add_argument("--out", default=None, help="Write JSON result to this path")
    p.add_argument("--hop", type=float, default=0.10, help="Hop size in seconds (default 0.10)")
    p.add_argument("--sr", type=int, default=22050, help="Target SR for analysis (default 22050)")
    p.add_argument("--min-seg", type=float, default=0.40, help="Minimum segment duration seconds")
    p.add_argument("--change-cost", type=float, default=2.0, help="Viterbi state change penalty")
    p.add_argument("--rms-threshold", type=float, default=0.01, help="RMS threshold for no-chord gating")

    args = p.parse_args(argv)

    result = transcribe_file(
        args.wav,
        target_sr=args.sr,
        hop_s=args.hop,
        min_seg_s=args.min_seg,
        change_cost=args.change_cost,
        rms_threshold=args.rms_threshold,
    )

    # Add a fingerprint for caching
    try:
        result["sha1"] = _sha1_file(args.wav)
    except Exception:
        result["sha1"] = None

    s = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(s, encoding="utf-8")
    else:
        print(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
