# Gemma 4 Lyrics Generator — Documentation

> **Version 2.0** · Chord-model-aware · Section-structured · Era-sensitive

---

## Overview

`gemma4_lyrics.py` is the lyrics generation stage of the ChordAI pipeline. It
consumes the token-string output of `sample_chords.py`, parses it into a
structured song specification, and uses **Gemma 4** (via Ollama) to write
complete, section-aware song lyrics.

The two tools share the same CLI flags so they can be run together as a single
pipeline with minimal glue code.

```
sample_chords.py  ──►  token string  ──►  gemma4_lyrics.py  ──►  lyrics (.txt + .json)
   [Chord Model]         (stdout)           [Lyrics Model]
```

---

## Installation

```bash
# Create a virtual environment (required on shared GPU servers)
python3 -m venv ~/ChordAI/venv
source ~/ChordAI/venv/bin/activate

# Install dependencies
pip install ollama rich

# Pull the model
ollama pull gemma4
```

---

## Data Format

### Input — Chord Token String

The lyrics model accepts the token string produced by `sample_chords.py`:

```
<bos> <genre:pop> <decade:2010> <intro> C <verse> F C E7 Amin <chorus> F C G C ... <eos>
```

| Token pattern      | Meaning                                     |
|--------------------|---------------------------------------------|
| `<bos>` / `<eos>`  | Start / end of sequence (ignored)           |
| `<genre:X>`        | Music genre — passed directly to the prompt |
| `<decade:X>`       | Era in 4-digit year e.g. `2010`             |
| `<intro>`, `<verse>`, `<chorus>`, `<bridge>`, `<outro>`, `<solo>`, `<prechorus>`, `<interlude>` | Section boundary markers |
| Any other token    | A chord symbol e.g. `F`, `Amin`, `E7`      |

### Output — Lyrics Files

For each run, two files are written to `--out_dir`:

**`lyrics_<genre>_<decade>_<timestamp>.txt`** — human-readable:
```
Genre  : POP
Decade : 2010s
Tempo  : 110 BPM
Vibe   : hopeful
Chords : <bos> <genre:pop> <decade:2010> <verse> F C G Am ...
──────────────────────────────────────────────────────────
[Verse 1]
(Chords: F C G Am)
...
```

**`lyrics_<genre>_<decade>_<timestamp>.json`** — machine-readable (for pipeline consumers):
```json
{
  "genre": "pop",
  "decade": 2010,
  "tempo_bpm": 110,
  "vibe": "hopeful",
  "sections": {
    "verse": ["F", "C", "G", "Am"],
    "chorus": ["F", "C", "G", "C"]
  },
  "raw_tokens": "<bos> <genre:pop> ...",
  "lyrics": "[Verse 1]\n..."
}
```

---

## CLI Reference

```
python gemma4_lyrics.py [OPTIONS]
```

### Chord-model-compatible flags
*(Mirror `sample_chords.py` exactly so the tools share the same interface)*

| Flag | Default | Description |
|------|---------|-------------|
| `--out_dir` | `out-chords` | Output directory, shared with chord model |
| `--mode` | `generate` | `generate` · `extend` · `section` |
| `--genre` | `pop` | Music genre passed to chord model |
| `--decade` | `2010` | Era in 4-digit year e.g. `1990` |
| `--seed_chords` | _(empty)_ | Seed progression for `extend`/`section` modes |
| `--next_section` | _(empty)_ | Target section name for `section` mode |

### Lyrics-specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tempo` | `110` | Tempo in BPM — shapes tone description in the prompt |
| `--vibe` | `emotional` | Free-text emotional vibe e.g. `"hopeful"`, `"world-weary"` |
| `--token_string` | _(empty)_ | Supply chord tokens directly, skipping the chord model |

### Model flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4` | Ollama model name |
| `--temp` | `0.85` | Sampling temperature (0.0 = deterministic, 1.0 = creative) |
| `--top_p` | `0.92` | Top-p / nucleus sampling threshold |
| `--no_save` | _(off)_ | Print lyrics to terminal only, write no files |

---

## Usage Examples

### Generate chords from scratch, then write lyrics
```bash
python gemma4_lyrics.py \
    --out_dir=out-chords \
    --mode=generate \
    --genre=pop \
    --decade=2010 \
    --tempo=110 \
    --vibe="hopeful"
```

### Extend a seed progression, then write lyrics
```bash
python gemma4_lyrics.py \
    --out_dir=out-chords \
    --mode=extend \
    --genre=rock \
    --decade=1990 \
    --seed_chords="Emin C G D" \
    --tempo=140 \
    --vibe="rebellious"
```

### Generate the next section of an existing song
```bash
python gemma4_lyrics.py \
    --out_dir=out-chords \
    --mode=section \
    --genre=pop \
    --decade=2010 \
    --seed_chords="<verse> C G Amin F C G Amin F" \
    --next_section=chorus \
    --tempo=95 \
    --vibe="melancholic"
```

### Bypass the chord model — supply token string directly
```bash
python gemma4_lyrics.py \
    --token_string="<bos> <genre:blues> <decade:1970> <verse> E A B7 E <chorus> A E B7 E <eos>" \
    --tempo=75 \
    --vibe="world-weary" \
    --no_save
```

---

## Prompt Engineering

The lyrics model uses a layered prompting strategy to produce genre-authentic,
era-accurate, harmonically-informed lyrics.

### Layer 1 — System prompt
Sets the model's persistent role as a professional songwriter with cross-genre
knowledge and music theory expertise.

### Layer 2 — Genre style guide
Each genre maps to a description of its **structural conventions**,
**vocabulary style**, and **overall mood**. Supported genres:

`blues` · `pop` · `country` · `rock` · `folk` · `jazz` · `rnb` · `metal` · `hiphop` · `edm`

Unknown genres fall through gracefully using the raw genre string.

### Layer 3 — Era context
The `--decade` value is used to inject a cultural and linguistic description
of the era into the prompt — e.g. `2010s` gets *"streaming era: vulnerable,
confessional, hook-first writing, social-media self-awareness"*.

### Layer 4 — Harmonic mood analysis
Chord symbols are scanned for structural indicators (`min`, `maj`, `7`, `dim`,
`sus`, `aug`, `add`, `9`) and translated into emotional language that is
injected into the prompt alongside the raw chord list.

### Layer 5 — Section-aware structure
Each section from the token string (`<verse>`, `<chorus>`, etc.) is listed
individually in the prompt with its own chord group. The model is instructed
to write a labelled section for each one in order.

### Layer 6 — Tempo mapping
The numeric `--tempo` BPM is converted to a descriptive feel:

| BPM range | Label |
|-----------|-------|
| < 70 | slow ballad |
| 70–99 | mid-tempo groove |
| 100–139 | driving up-tempo |
| ≥ 140 | fast and intense |

### Layer 7 — Vibe threading
The free-text `--vibe` string (e.g. `"melancholic"`, `"hopeful"`) is woven
into the prompt in two places: once in the parameters block and once in the
format rules, instructing the model to let the vibe *evolve naturally* across
the song rather than repeat it mechanically.

---

## Architecture & Pipeline Integration

```
┌─────────────────────────────────────────────────────────┐
│                     pipeline.py (future)                │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
                ▼                         ▼
  ┌─────────────────────┐     ┌─────────────────────────┐
  │   sample_chords.py  │────►│    gemma4_lyrics.py     │
  │   (Chord Model)     │     │    (Lyrics Model)       │
  │                     │     │                         │
  │  stdout: token str  │     │  in:  token string      │
  │  file:  out-chords/ │     │  out: .txt + .json      │
  └─────────────────────┘     └─────────────────────────┘
```

### SongSpec — the shared data contract

Both models communicate via the `SongSpec` dataclass:

```python
@dataclass
class SongSpec:
    genre:     str               # e.g. "pop"
    decade:    int               # e.g. 2010
    tempo_bpm: int               # e.g. 110
    vibe:      str               # e.g. "hopeful"
    sections:  dict[str, list[str]]  # e.g. {"verse": ["F","C","G"], ...}
    raw_tokens: str              # original token string (preserved for audit)
```

The `.json` output file serialises this struct, making it easy for any future
downstream model (melody generation, arrangement, mixing) to consume the data
without re-parsing the token string.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Missing dependency` | `pip install ollama rich` |
| `externally-managed-environment` | Use a venv: `python3 -m venv venv && source venv/bin/activate` |
| `Model 'gemma4' not found` | `ollama pull gemma4` |
| `Ollama unreachable` | `ollama serve` (in a separate terminal or as a background service) |
| `No .txt output found in out-chords` | Check that `sample_chords.py` ran successfully; inspect stderr |
| Token string not detected in stdout | The chord model may write to a file — ensure `--out_dir` is correct |

---

## Supported Genres

| Genre | Era notes |
|-------|-----------|
| `blues` | Works best with `--decade=1950` through `1970` |
| `pop` | Decade-sensitive — 1980s vs 2010s produce very different results |
| `rock` | `--decade=1990` with high `--tempo` recommended for grunge/Britpop |
| `folk` | Lower tempos (60–90 BPM), introspective vibes |
| `jazz` | Works well with extended chords (7ths, 9ths) in seed progressions |
| `country` | `--decade=1970` through `1990` for classic Nashville sound |
| `rnb` | `--decade=2000` or `2010` for modern R&B phrasing |
| `metal` | High tempo (140+), dark vibes, `--decade=1980` or `1990` |
| `hiphop` | `--decade=1990` for boom-bap, `2010` for trap influence |
| `edm` | Best paired with `section` mode to target drops and build-ups |