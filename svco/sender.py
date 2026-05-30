"""
Sends a completed SVCO session payload to the ChordAI Go server,
then forwards the SongResult to the Nuxt frontend via /api/voice-result
so it appears in the chat UI.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

log = logging.getLogger(__name__)

# Go backend — override with CHORDAI_SERVER env var
DEFAULT_SERVER = "http://localhost:5555"

# Nuxt frontend — override with CHORDAI_FRONTEND env var
DEFAULT_FRONTEND = "http://localhost:3000"


def _server_url() -> str:
    return os.environ.get("CHORDAI_SERVER", DEFAULT_SERVER).rstrip("/")


def _frontend_url() -> str:
    return os.environ.get("CHORDAI_FRONTEND", DEFAULT_FRONTEND).rstrip("/")


import re as _re


def _to_model_vocab(chord: str) -> str:
    """
    Convert an offline_chords display label to the nanoGPT chord model vocabulary.

    offline_chords emits:  Am  C#m  F#  G7  Csus4
    model vocab uses:      Amin  Csmin  Fs  G7  Csus4

    Rules:
    - Replace '#' with 's'  (C# → Cs, F# → Fs)
    - Expand trailing 'm' to 'min' when it immediately follows a note name
      (Am → Amin, Csm → Csmin) but leave m in the middle alone (Cmaj7 untouched)
    """
    chord = chord.replace("#", "s")
    # Trailing 'm' that ends the string after a root/modifier character
    chord = _re.sub(r"m$", "min", chord)
    return chord


def build_planner_input(user_speech: str, user_chords: list[Any]) -> dict:
    """
    Convert SVCO session output into a PlannerInput dict.

    user_speech  — Whisper transcript of what the user said
    user_chords  — list of chord segment dicts from offline_chords.transcribe_file,
                   each with a "display" key like "Am", "G", "C"
    """
    chord_names = [
        _to_model_vocab(seg["display"])
        for seg in user_chords
        if isinstance(seg, dict) and seg.get("display") not in (None, "N")
    ]

    deduped: list[str] = []
    for chord in chord_names:
        if not deduped or deduped[-1] != chord:
            deduped.append(chord)

    seed_chords = " ".join(deduped)
    mode = "extend" if seed_chords else "generate"

    return {
        "freetext": user_speech.strip(),
        "seed_chords": seed_chords,
        "mode": mode,
    }


def _post_json(url: str, data: dict, *, timeout: int) -> dict | None:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:
        log.error("[sender] POST %s failed: %s", url, exc)
        return None
    except json.JSONDecodeError as exc:
        log.error("[sender] could not parse response from %s: %s", url, exc)
        return None


def send_to_server(
    user_speech: str,
    user_chords: list[Any],
    *,
    timeout: int = 120,
) -> dict | None:
    """
    1. POST { type: 'start' } to Nuxt → browser shows loading state immediately.
    2. POST PlannerInput to Go server /api/generate → receive SongResult.
    3. POST { type: 'result', ...SongResult } to Nuxt → browser fills in the result.

    Returns the SongResult on success, None on failure.
    """
    payload = build_planner_input(user_speech, user_chords)
    go_url = f"{_server_url()}/api/generate"

    # Notify the frontend immediately so the user sees a loading state
    frontend_url = f"{_frontend_url()}/api/voice-result"
    _post_json(frontend_url, {"type": "start", "freetext": payload.get("freetext", "")}, timeout=5)

    log.info("[sender] → Go server %s  freetext=%r seed_chords=%r", go_url, payload.get("freetext", "")[:60], payload.get("seed_chords", ""))

    result = _post_json(go_url, payload, timeout=timeout)
    if result is None:
        log.warning("[sender] Go server returned no result.")
        return None

    log.info("[sender] SongResult: genre=%s decade=%s", result.get("genre"), result.get("decade"))

    # Forward result to Nuxt frontend — wrapped with type:'result' so the
    # frontend can distinguish it from the earlier type:'start' notification
    log.info("[sender] → Nuxt frontend %s", frontend_url)
    pushed = _post_json(frontend_url, {"type": "result", **result}, timeout=10)
    if pushed is None:
        log.warning("[sender] Could not push to frontend — is Nuxt running?")
    else:
        log.info("[sender] Result delivered to frontend.")

    return result
