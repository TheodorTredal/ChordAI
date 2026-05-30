"""
Shared Pydantic schemas for the ChordAI pipeline.

These mirror the Go structs in server/schemas/schemas.go exactly.
Field names and JSON keys must stay in sync with the Go json tags.

Usage in any model script:
    from schemas import PlannerInput, PlannerDecision, SongSpec, SongResult
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PlannerInput(BaseModel):
    """What the client sends to the Go server — and what SVCO should produce."""
    freetext: str = ""
    genre: Optional[str] = None
    decade: Optional[int] = None
    tempo_bpm: Optional[int] = None
    vibe: Optional[str] = None
    mode: Optional[str] = None          # "generate" | "extend" | "section"
    seed_chords: str = ""
    next_section: str = ""

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("generate", "extend", "section"):
            raise ValueError(f"mode must be generate | extend | section, got {v!r}")
        return v

    @field_validator("tempo_bpm")
    @classmethod
    def validate_tempo(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (40 <= v <= 200):
            raise ValueError(f"tempo_bpm must be 40–200, got {v}")
        return v

    @field_validator("decade")
    @classmethod
    def validate_decade(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 10 != 0:
            raise ValueError(f"decade must be a multiple of 10, got {v}")
        return v


class PlannerDecision(BaseModel):
    """What the Ollama planner (llama3.2) returns — mirrors Go PlannerDecision."""
    genre: str
    decade: int
    tempo_bpm: int
    vibe: str
    mode: str = "generate"
    seed_chords: str = ""
    next_section: str = ""
    pipeline: list[str] = Field(default_factory=lambda: ["chord_model", "lyrics_model"])
    reasoning: str = ""

    VALID_PIPELINE_STAGES = {"chord_model", "lyrics_model", "image_model"}

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline(cls, v: list[str]) -> list[str]:
        unknown = set(v) - cls.VALID_PIPELINE_STAGES
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {unknown}. Valid: {cls.VALID_PIPELINE_STAGES}")
        if not v:
            raise ValueError("pipeline must contain at least one stage")
        return v

    @field_validator("tempo_bpm")
    @classmethod
    def validate_tempo(cls, v: int) -> int:
        if not (40 <= v <= 200):
            raise ValueError(f"tempo_bpm must be 40–200, got {v}")
        return v

    @field_validator("decade")
    @classmethod
    def validate_decade(cls, v: int) -> int:
        clamped = max(1950, min(2020, (v // 10) * 10))
        return clamped

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("generate", "extend", "section"):
            raise ValueError(f"mode must be generate | extend | section, got {v!r}")
        return v


class SongSpec(BaseModel):
    """
    Shared contract between the chord model and the lyrics model.
    Mirrors Go SongSpec. This is what chord_runner produces and
    lyrics_runner consumes.
    """
    genre: str
    decade: int
    tempo_bpm: int
    vibe: str
    sections: dict[str, list[str]] = Field(default_factory=dict)
    raw_tokens: str = ""

    @property
    def all_chords(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for chords in self.sections.values():
            for c in chords:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    @property
    def structure_label(self) -> str:
        return " / ".join(s.capitalize() for s in self.sections)


class SongResult(BaseModel):
    """
    Final output returned to the client.
    Mirrors Go SongResult — all field names must match the Go json tags.
    """
    genre: str
    decade: int
    tempo_bpm: int
    vibe: str
    sections: dict[str, list[str]] = Field(default_factory=dict)
    raw_tokens: str = ""
    lyrics: str = ""
    image_path: Optional[str] = None


class WSEvent(BaseModel):
    """WebSocket event streamed to the client during pipeline execution."""
    stage: str
    status: str     # "running" | "done" | "streaming" | "error"
    token: Optional[str] = None
    error: Optional[str] = None
