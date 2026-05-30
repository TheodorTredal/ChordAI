#!/usr/bin/env python3
"""
🎵 Gemma 4 Song Lyrics Generator
─────────────────────────────────────────────────────────────────────────────
Consumes the token-string output of sample_chords.py and generates
section-aware lyrics using Gemma 4 via Ollama.

The chord model emits tokens in this format:
  <bos> <genre:pop> <decade:2010> <intro> C <verse> F C E7 Amin <chorus> F C G C ... <eos>

This script mirrors the chord model's CLI flags so the two tools share the
same interface, adding --tempo and --vibe as lyrics-specific extras.

Requirements:
    pip install ollama rich          (or use a venv — see README)
    ollama pull gemma4

Modes (mirror sample_chords.py):
    generate   Run chord model from scratch, then generate lyrics
    extend     Extend a seed progression, then generate lyrics
    section    Generate the next section of chords, then lyrics

Usage:
    python gemma4_lyrics.py --out_dir=out-chords --mode=generate \\
        --genre=pop --decade=2010 --tempo=110 --vibe="hopeful"

    python gemma4_lyrics.py --out_dir=out-chords --mode=extend \\
        --genre=rock --decade=1990 --seed_chords="Emin C G D" \\
        --tempo=140 --vibe="rebellious"

    python gemma4_lyrics.py --out_dir=out-chords --mode=section \\
        --genre=pop --decade=2010 --tempo=95 --vibe="melancholic" \\
        --seed_chords="<verse> C G Amin F C G Amin F" --next_section=chorus
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import ollama
except ImportError:
    print("Missing dependency. Run:  pip install ollama rich pydantic")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
except ImportError:
    print("Missing dependency. Run:  pip install ollama rich pydantic")
    sys.exit(1)

# Import shared schemas — same field names as Go server/schemas/schemas.go
sys.path.insert(0, str(Path(__file__).parent))
from schemas import SongSpec

# Rich console writes to stderr so stdout stays clean for machine-readable output
console = Console(stderr=True)

KNOWN_SECTIONS = ["intro", "verse", "prechorus", "chorus", "bridge", "outro", "solo", "interlude"]


# ─────────────────────────────────────────────────────────────────────────────
# Token-string parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_chord_token_string(token_str: str, tempo_bpm: int, vibe: str) -> SongSpec:
    """
    Parse the chord model's token string into a SongSpec.

    Expected format:
      <bos> <genre:pop> <decade:2010> <intro> C <verse> F C E7 Amin <chorus> F C G C ... <eos>
    """
    tokens = token_str.strip().split()

    genre  = "pop"
    decade = 2010
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for tok in tokens:
        if tok in ("<bos>", "<eos>"):
            continue

        # <genre:X>
        m = re.fullmatch(r"<genre:(.+?)>", tok)
        if m:
            genre = m.group(1).lower()
            continue

        # <decade:X>
        m = re.fullmatch(r"<decade:(\d+)>", tok)
        if m:
            decade = int(m.group(1))
            continue

        # section tag  <verse>, <chorus>, etc.
        m = re.fullmatch(r"<(\w+)>", tok)
        if m and m.group(1) in KNOWN_SECTIONS:
            current_section = m.group(1)
            if current_section not in sections:
                sections[current_section] = []
            continue

        # plain chord token
        if current_section is not None and tok:
            sections[current_section].append(tok)

    return SongSpec(
        genre=genre,
        decade=decade,
        tempo_bpm=tempo_bpm,
        vibe=vibe,
        sections=sections,
        raw_tokens=token_str.strip(),
    )  # type: ignore[call-arg]  # Pydantic accepts kwargs


def load_latest_token_file(out_dir: str) -> str:
    """
    Fall-back: if the chord model writes to a file instead of stdout,
    pick the most recently modified .txt file in out_dir.
    """
    p = Path(out_dir)
    files = sorted(p.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No .txt output found in {out_dir!r}")
    return files[0].read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Chord model runner
# ─────────────────────────────────────────────────────────────────────────────

def run_chord_model(args: argparse.Namespace) -> str:
    """
    Execute sample_chords.py with the matching flags and capture its stdout.
    Falls back to scanning out_dir for a freshly written file.
    """
    cmd = [
        sys.executable, "sample_chords.py",
        f"--out_dir={args.out_dir}",
        f"--mode={args.mode}",
        f"--genre={args.genre}",
        f"--decade={args.decade}",
    ]
    if args.seed_chords:
        cmd.append(f"--seed_chords={args.seed_chords}")
    if args.next_section:
        cmd.append(f"--next_section={args.next_section}")

    console.print(f"\n[bold blue]▶  Running chord model:[/bold blue] {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]✗ Chord model exited with code {result.returncode}[/red]")
        console.print(result.stderr)
        sys.exit(1)

    stdout = result.stdout.strip()

    # Try to find the token string in stdout first
    for line in stdout.splitlines():
        if "<bos>" in line or "<genre:" in line:
            return line.strip()

    # Fall back to reading from out_dir
    console.print("[dim]Token string not found in stdout — reading from out_dir...[/dim]")
    return load_latest_token_file(args.out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Era / decade style context
# ─────────────────────────────────────────────────────────────────────────────

DECADE_CONTEXT = {
    1950: "Rock and roll era: simple innocent language, themes of cars, dances, young love",
    1960: "Counterculture era: protest, psychedelia, poetic imagery, social change",
    1970: "Singer-songwriter / disco era: personal introspection, groove, escapism",
    1980: "Synth-pop / power ballad era: big emotions, neon imagery, anthemic feel",
    1990: "Grunge / Britpop era: raw authenticity, irony, disillusionment, lo-fi attitude",
    2000: "Post-millennial: digital age anxiety, polished pop, introspective hip-hop influence",
    2010: "Streaming era: vulnerable, confessional, hook-first writing, social-media self-awareness",
    2020: "Bedroom-pop era: intimate production aesthetic, pandemic themes, identity, mental health",
}

def decade_context(decade: int) -> str:
    rounded = (decade // 10) * 10
    return DECADE_CONTEXT.get(rounded, f"{decade}s aesthetic and cultural references")


# ─────────────────────────────────────────────────────────────────────────────
# Genre style table
# ─────────────────────────────────────────────────────────────────────────────

GENRE_STYLES = {
    "blues":   {"description": "12-bar call-and-response, raw hardship and perseverance", "vocabulary": "gritty, repetition of key lines, AAB rhyme", "mood": "melancholic yet resilient"},
    "pop":     {"description": "verse-prechorus-chorus-bridge, catchy hooks", "vocabulary": "upbeat, modern, memorable one-liners, ABAB rhyme", "mood": "energetic, emotional"},
    "country": {"description": "storytelling verses, singalong chorus, vivid imagery", "vocabulary": "plain-spoken, everyday life — roads, family, heartbreak", "mood": "nostalgic, warm, honest"},
    "rock":    {"description": "punchy verse-chorus, anthemic energy", "vocabulary": "bold, rebellious, visceral imagery", "mood": "intense, rebellious"},
    "folk":    {"description": "narrative poetry, acoustic imagery, personal/political", "vocabulary": "literary, metaphor-rich, flowing lines", "mood": "introspective, organic"},
    "jazz":    {"description": "impressionistic, conversational swing", "vocabulary": "elegant, urbane, clever wordplay", "mood": "cool, smoky, bittersweet"},
    "rnb":     {"description": "soulful groove, love and vulnerability", "vocabulary": "sensual, emotionally direct, modern slang", "mood": "intimate, passionate"},
    "metal":   {"description": "aggressive verse-chorus, dark or epic themes", "vocabulary": "dramatic, epic scale — battle, darkness, defiance", "mood": "powerful, cathartic"},
    "hiphop":  {"description": "bar-based verses, hook chorus, storytelling flow", "vocabulary": "rhythmically dense, wordplay, internal rhyme, vernacular", "mood": "confident, raw, reflective"},
    "edm":     {"description": "build-drop-breakdown structure, euphoric hooks", "vocabulary": "repetitive, hypnotic, emotion-forward, minimal narrative", "mood": "euphoric, transcendent"},
}

CHORD_MOOD_HINTS = {
    "min":  "melancholic, tense, emotional",
    "maj":  "bright, uplifting, resolved",
    "7":    "bluesy, unresolved tension",
    "dim":  "sinister, uneasy",
    "aug":  "dreamlike, unstable",
    "sus":  "floating, unresolved, searching",
    "add":  "colourful, open, modern",
    "9":    "lush, jazzy, sophisticated",
}

def analyse_chords(chords: list[str]) -> str:
    hints = []
    chord_str = " ".join(chords).lower()
    for keyword, hint in CHORD_MOOD_HINTS.items():
        if keyword in chord_str:
            hints.append(hint)
    return "; ".join(dict.fromkeys(hints)) or "balanced, versatile emotional range"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return (
        "You are an experienced songwriter with deep knowledge of music history, "
        "music theory, and lyric craft across all major genres and eras. "
        "You write lyrics that feel authentically tied to the chords, era, and emotional vibe provided. "
        "You never write placeholder text — always complete, polished, singable lyrics."
    )


def build_user_prompt(spec: SongSpec) -> str:
    style = GENRE_STYLES.get(spec.genre.lower(), {
        "description": f"{spec.genre} style",
        "vocabulary": "fitting the genre",
        "mood": "appropriate to the genre",
    })
    chord_mood  = analyse_chords(spec.all_chords)
    era_context = decade_context(spec.decade)

    # Build per-section chord annotations
    section_breakdown = "\n".join(
        f"  [{s.upper()}]  →  {' '.join(chords)}"
        for s, chords in spec.sections.items()
    ) or f"  All sections: {' '.join(spec.all_chords)}"

    tempo_feel = (
        "slow ballad"      if spec.tempo_bpm < 70  else
        "mid-tempo groove" if spec.tempo_bpm < 100 else
        "driving up-tempo" if spec.tempo_bpm < 140 else
        "fast and intense"
    )

    return f"""Write complete, original song lyrics. Output ONLY the lyrics — no introduction, commentary, or meta-text.

═══════════════════════════════════════════════════
SONG PARAMETERS
═══════════════════════════════════════════════════
Genre       : {spec.genre.upper()}
Era / decade: {spec.decade}s
Tempo       : {spec.tempo_bpm} BPM ({tempo_feel})
Emotional vibe: {spec.vibe}

═══════════════════════════════════════════════════
CHORD STRUCTURE  (one group per song section)
═══════════════════════════════════════════════════
{section_breakdown}

Harmonic mood suggested by these chords:
  {chord_mood}

═══════════════════════════════════════════════════
ERA CONTEXT
═══════════════════════════════════════════════════
{era_context}
Reflect this era authentically in vocabulary, imagery, and cultural references.

═══════════════════════════════════════════════════
GENRE STYLE GUIDE
═══════════════════════════════════════════════════
Structure style : {style['description']}
Vocabulary/style: {style['vocabulary']}
Overall mood    : {style['mood']}

═══════════════════════════════════════════════════
FORMAT RULES
═══════════════════════════════════════════════════
- Write a section for EVERY section listed in CHORD STRUCTURE above, in order.
- Label each section: [Intro], [Verse 1], [Chorus], [Bridge], etc.
- Annotate chords at the top of each section in parentheses, e.g.: (Chords: F C G C)
- Each verse: 4–6 lines. Chorus: 4–6 lines, memorable and repeatable.
- Match the emotional vibe "{spec.vibe}" throughout — let it evolve naturally.
- Do NOT write a title, explanation, or any text outside the labelled sections.

Write the song now:"""


# ─────────────────────────────────────────────────────────────────────────────
# Ollama generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_lyrics(spec: SongSpec, model: str, temperature: float, top_p: float) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_prompt(spec)},
    ]

    console.print("\n[bold cyan]✍  Generating lyrics...[/bold cyan]\n")
    full_response: list[str] = []

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": temperature, "top_p": top_p, "num_predict": 1500},
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        full_response.append(token)
        console.print(token, end="", highlight=False)

    console.print()
    return "".join(full_response)


# ─────────────────────────────────────────────────────────────────────────────
# Output saving
# ─────────────────────────────────────────────────────────────────────────────

def save_output(spec: SongSpec, lyrics: str, out_dir: str) -> Path:
    """Save lyrics + metadata JSON to out_dir and return the JSON path."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base  = p / f"lyrics_{spec.genre}_{spec.decade}_{stamp}"

    # Plain text (human-readable copy)
    txt_path = base.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Genre  : {spec.genre.upper()}\n")
        f.write(f"Decade : {spec.decade}s\n")
        f.write(f"Tempo  : {spec.tempo_bpm} BPM\n")
        f.write(f"Vibe   : {spec.vibe}\n")
        f.write(f"Chords : {spec.raw_tokens}\n")
        f.write("─" * 50 + "\n\n")
        f.write(lyrics)

    # Metadata JSON — read by Go server via stdout path (see main())
    meta_path = base.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "genre":      spec.genre,
            "decade":     spec.decade,
            "tempo_bpm":  spec.tempo_bpm,
            "vibe":       spec.vibe,
            "sections":   spec.sections,
            "raw_tokens": spec.raw_tokens,
            "lyrics":     lyrics,
        }, f, indent=2)

    return meta_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (mirrors sample_chords.py flags + adds --tempo and --vibe)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate section-aware song lyrics from chord model output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate chords from scratch, then write lyrics
  python gemma4_lyrics.py --out_dir=out-chords --mode=generate \\
      --genre=pop --decade=2010 --tempo=110 --vibe="hopeful"

  # Extend a seed progression, then write lyrics
  python gemma4_lyrics.py --out_dir=out-chords --mode=extend \\
      --genre=rock --decade=1990 --seed_chords="Emin C G D" \\
      --tempo=140 --vibe="rebellious"

  # Generate the next section of an existing song
  python gemma4_lyrics.py --out_dir=out-chords --mode=section \\
      --genre=pop --decade=2010 --tempo=95 --vibe="melancholic" \\
      --seed_chords="<verse> C G Amin F C G Amin F" --next_section=chorus

  # Skip the chord model — supply a token string directly
  python gemma4_lyrics.py --token_string="<bos> <genre:blues> <decade:1970> <verse> E A B7 E <eos>" \\
      --tempo=75 --vibe="world-weary"
        """,
    )

    # ── Chord-model-compatible flags ──────────────────────────────────────────
    p.add_argument("--out_dir",      default="out-chords",  help="Output directory (shared with chord model)")
    p.add_argument("--mode",         default="generate",    choices=["generate", "extend", "section"],
                   help="Chord generation mode")
    p.add_argument("--genre",        default="pop",         help="Music genre")
    p.add_argument("--decade",       type=int, default=2010,help="Decade (e.g. 1990, 2010)")
    p.add_argument("--seed_chords",  default="",            help="Seed chord string (extend/section modes)")
    p.add_argument("--next_section", default="",            help="Target section name (section mode)")

    # ── Lyrics-specific flags ─────────────────────────────────────────────────
    p.add_argument("--tempo",        type=int, default=110, help="Tempo in BPM (default: 110)")
    p.add_argument("--vibe",         default="emotional",   help='Emotional vibe, e.g. "hopeful", "melancholic"')
    p.add_argument("--token_string", default="",            help="Supply chord token string directly (skips chord model)")

    # ── Model flags ───────────────────────────────────────────────────────────
    p.add_argument("--model",        default="gemma4",      help="Ollama model name (default: gemma4)")
    p.add_argument("--temp",         type=float, default=0.85, help="Sampling temperature 0.0–1.0")
    p.add_argument("--top_p",        type=float, default=0.92, help="Top-p sampling (default: 0.92)")
    p.add_argument("--no_save",      action="store_true",   help="Print lyrics only, do not save files")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    console.print(Panel.fit(
        "[bold magenta]🎵  Gemma 4 Lyrics Generator[/bold magenta]\n"
        "[dim]Chord-model-aware · Section-structured · Era-sensitive[/dim]",
        border_style="magenta",
    ))

    # ── Step 1: get the chord token string ───────────────────────────────────
    if args.token_string:
        token_str = args.token_string
        console.print("[dim]Using supplied --token_string[/dim]")
    else:
        token_str = run_chord_model(args)

    console.print(f"\n[bold]Token string:[/bold] [dim]{token_str}[/dim]")

    # ── Step 2: parse into SongSpec ──────────────────────────────────────────
    spec = parse_chord_token_string(token_str, tempo_bpm=args.tempo, vibe=args.vibe)

    console.print(Panel(
        f"[bold]Genre  :[/bold] {spec.genre.upper()}\n"
        f"[bold]Decade :[/bold] {spec.decade}s\n"
        f"[bold]Tempo  :[/bold] {spec.tempo_bpm} BPM\n"
        f"[bold]Vibe   :[/bold] {spec.vibe}\n"
        f"[bold]Sections:[/bold] {spec.structure_label}\n"
        f"[bold]Chords :[/bold] {' '.join(spec.all_chords)}",
        title="[yellow]Song Spec[/yellow]",
        border_style="yellow",
    ))

    # ── Step 3: verify Ollama ────────────────────────────────────────────────
    try:
        ollama.show(args.model)
    except ollama.ResponseError:
        console.print(f"[red]✗ Model '{args.model}' not found.[/red]  Run: ollama pull {args.model}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Ollama unreachable: {e}[/red]  Run: ollama serve")
        sys.exit(1)

    # ── Step 4: generate ────────────────────────────────────────────────────
    lyrics = generate_lyrics(spec, model=args.model, temperature=args.temp, top_p=args.top_p)

    # ── Step 5: save ─────────────────────────────────────────────────────────
    if not args.no_save:
        saved = save_output(spec, lyrics, args.out_dir)
        console.print(f"\n[green]✓ Saved:[/green] {saved}  [dim](+ .txt copy)[/dim]", file=sys.stderr)
        # Print the JSON path to stdout as the last line — Go server reads this.
        print(str(saved), flush=True)

    console.print("\n[bold green]✓ Done![/bold green]", file=sys.stderr)


if __name__ == "__main__":
    main()