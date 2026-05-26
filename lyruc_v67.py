#!/usr/bin/env python3
"""
🎵 Gemma 4 Song Lyrics Generator
Generates song lyrics from a chord progression and genre using Gemma 4 via Ollama.

Requirements:
    pip install ollama rich
    ollama pull gemma4

Usage:
    python gemma4_lyrics.py
    python gemma4_lyrics.py --chords "Am F C G" --genre blues --theme "lost love"
"""

import argparse
import sys

try:
    import ollama
except ImportError:
    print("Missing dependency. Run:  pip install ollama rich")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.text import Text
    from rich import print as rprint
except ImportError:
    print("Missing dependency. Run:  pip install ollama rich")
    sys.exit(1)

console = Console()

# ---------------------------------------------------------------------------
# Genre-specific prompt guidance
# ---------------------------------------------------------------------------

GENRE_STYLES = {
    "blues": {
        "description": "12-bar blues structure with call-and-response lines, raw emotion, themes of hardship and perseverance",
        "vocabulary": "gritty, visceral, repetition of key lines, simple rhymes (AABB or AAB)",
        "mood": "melancholic yet resilient",
    },
    "pop": {
        "description": "verse-pre-chorus-chorus-bridge structure with catchy hooks and relatable themes",
        "vocabulary": "upbeat, modern, memorable one-liners, strong rhyme scheme (ABAB or ABCB)",
        "mood": "energetic, emotional, broadly relatable",
    },
    "country": {
        "description": "storytelling-focused with vivid imagery, narrative verses and singalong chorus",
        "vocabulary": "conversational, plain-spoken, rooted in everyday life — trucks, roads, family, heartbreak",
        "mood": "nostalgic, warm, honest",
    },
    "rock": {
        "description": "punchy verse-chorus structure with anthemic choruses and edgy attitude",
        "vocabulary": "bold, rebellious, visceral imagery, driving rhythm in word choices",
        "mood": "intense, rebellious, powerful",
    },
    "folk": {
        "description": "narrative poetry set to music, acoustic imagery, personal or political themes",
        "vocabulary": "literary, metaphor-rich, flowing lines that mirror fingerpicking rhythms",
        "mood": "introspective, organic, timeless",
    },
    "jazz": {
        "description": "sophisticated, impressionistic lyrics with complex imagery and a conversational swing",
        "vocabulary": "elegant, urbane, clever wordplay, crooner-style phrasing",
        "mood": "cool, smoky, bittersweet",
    },
    "rnb": {
        "description": "smooth, soulful lyrics about love, vulnerability and empowerment with groove-driven phrasing",
        "vocabulary": "sensual, emotionally direct, modern slang and vocal ad-libs in brackets",
        "mood": "intimate, passionate, soulful",
    },
    "metal": {
        "description": "aggressive verse-chorus with dark or epic themes, heavy imagery and power",
        "vocabulary": "dramatic, intense, epic scale — battle, darkness, defiance",
        "mood": "powerful, cathartic, darkly epic",
    },
}

CHORD_MOOD_HINTS = {
    "minor": "melancholic, tense, emotional",
    "major": "bright, uplifting, resolved",
    "dominant7": "bluesy, unresolved tension",
    "diminished": "sinister, uneasy",
    "augmented": "dreamlike, unstable",
    "suspended": "floating, unresolved, searching",
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def analyse_chords(chords: list[str]) -> str:
    """Return a short mood hint based on chord names."""
    hints = []
    chord_str = " ".join(chords).lower()
    for keyword, hint in CHORD_MOOD_HINTS.items():
        if keyword[:3] in chord_str or (keyword == "minor" and "m" in chord_str):
            hints.append(hint)
    if not hints:
        hints.append("balanced, versatile emotional range")
    return "; ".join(dict.fromkeys(hints))  # deduplicate, preserve order


def build_system_prompt() -> str:
    return (
        "You are an experienced songwriter and lyricist with deep knowledge of music theory "
        "and multiple genres. You understand how chord progressions shape the emotional "
        "landscape of a song, and you craft lyrics that feel naturally linked to the harmony "
        "beneath them. You always write complete, polished lyrics — never placeholders."
    )


def build_user_prompt(
    chords: list[str],
    genre: str,
    theme: str,
    structure: str,
    tempo_feel: str,
) -> str:
    style = GENRE_STYLES.get(genre.lower(), {
        "description": f"{genre} style",
        "vocabulary": "fitting the genre",
        "mood": "appropriate to the genre",
    })
    chord_mood = analyse_chords(chords)
    chord_display = " → ".join(chords)

    return f"""Write complete, original song lyrics for the following song. Output ONLY the lyrics — no introductions, no commentary, no meta-text.

═══════════════════════════════════════
SONG PARAMETERS
═══════════════════════════════════════
Chord progression : {chord_display}
Genre             : {genre.upper()}
Theme / subject   : {theme}
Song structure    : {structure}
Tempo feel        : {tempo_feel}

═══════════════════════════════════════
HARMONIC MOOD GUIDANCE
═══════════════════════════════════════
The chords suggest: {chord_mood}
Use this emotional palette to shape the tone, word choices and imagery of the lyrics.

═══════════════════════════════════════
GENRE STYLE GUIDE
═══════════════════════════════════════
Structure style : {style['description']}
Vocabulary/style: {style['vocabulary']}
Overall mood    : {style['mood']}

═══════════════════════════════════════
FORMAT RULES
═══════════════════════════════════════
- Label each section clearly: [Verse 1], [Chorus], [Bridge], etc.
- Each verse: 4–6 lines
- Chorus: 4–6 lines, memorable and repeatable
- Include a bridge if the structure calls for it
- Annotate the chord progression at the start of each section like this:
  (Chords: {chord_display})
- Do NOT include any explanation, title, or notes — output only the labelled lyrics.

Write the song now:"""


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def generate_lyrics(
    chords: list[str],
    genre: str,
    theme: str,
    structure: str,
    tempo_feel: str,
    model: str = "gemma4",
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> str:
    """Stream lyrics from Gemma 4 via Ollama and return the full text."""

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_prompt(chords, genre, theme, structure, tempo_feel)},
    ]

    full_response = []
    console.print("\n[bold cyan]✍  Generating lyrics...[/bold cyan]\n")

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": 1024,
        },
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        full_response.append(token)
        console.print(token, end="", highlight=False)

    console.print()  # newline after stream
    return "".join(full_response)


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

STRUCTURES = [
    "Verse / Chorus / Verse / Chorus / Bridge / Chorus",
    "Verse / Verse / Chorus / Verse / Chorus",
    "Verse / Pre-Chorus / Chorus / Verse / Pre-Chorus / Chorus / Bridge / Chorus",
    "12-bar blues (three verses)",
    "Through-composed (no repeated sections)",
]

TEMPO_FEELS = ["slow ballad", "mid-tempo groove", "up-tempo driving", "swung / shuffle", "rubato (free time)"]


def interactive_mode() -> dict:
    """Walk the user through inputs interactively."""
    console.print(Panel.fit(
        "[bold magenta]🎸 Gemma 4 Lyrics Generator[/bold magenta]\n"
        "[dim]Powered by Ollama + Gemma 4[/dim]",
        border_style="magenta",
    ))

    # Chords
    console.print("\n[bold]Enter your chord progression[/bold] [dim](e.g. Am F C G  or  Dm7 G7 Cmaj7 Am7)[/dim]")
    chords_raw = Prompt.ask("[green]Chords[/green]", default="Am F C G")
    chords = [c.strip() for c in chords_raw.replace(",", " ").split() if c.strip()]

    # Genre
    genres = list(GENRE_STYLES.keys()) + ["other"]
    console.print(f"\n[bold]Available genres:[/bold] {', '.join(genres)}")
    genre = Prompt.ask("[green]Genre[/green]", default="pop")

    # Theme
    theme = Prompt.ask("\n[green]Theme / subject[/green] [dim](e.g. 'lost love', 'chasing dreams', 'hometown nostalgia')[/dim]",
                       default="lost love")

    # Structure
    console.print("\n[bold]Song structures:[/bold]")
    for i, s in enumerate(STRUCTURES, 1):
        console.print(f"  [cyan]{i}[/cyan]. {s}")
    struct_choice = Prompt.ask("[green]Choose structure[/green] (1-5)", default="1")
    try:
        structure = STRUCTURES[int(struct_choice) - 1]
    except (ValueError, IndexError):
        structure = STRUCTURES[0]

    # Tempo
    console.print("\n[bold]Tempo feels:[/bold]")
    for i, t in enumerate(TEMPO_FEELS, 1):
        console.print(f"  [cyan]{i}[/cyan]. {t}")
    tempo_choice = Prompt.ask("[green]Choose tempo feel[/green] (1-5)", default="2")
    try:
        tempo_feel = TEMPO_FEELS[int(tempo_choice) - 1]
    except (ValueError, IndexError):
        tempo_feel = TEMPO_FEELS[1]

    return dict(chords=chords, genre=genre, theme=theme, structure=structure, tempo_feel=tempo_feel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate song lyrics with Gemma 4 via Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemma4_lyrics.py
  python gemma4_lyrics.py --chords "Dm G C Am" --genre folk --theme "leaving home"
  python gemma4_lyrics.py --chords "E A B7" --genre blues --theme "hard times" --model gemma4:27b
        """,
    )
    parser.add_argument("--chords",   type=str, help='Chord progression e.g. "Am F C G"')
    parser.add_argument("--genre",    type=str, help=f"Genre: {', '.join(GENRE_STYLES.keys())}")
    parser.add_argument("--theme",    type=str, default="lost love", help="Song theme or subject")
    parser.add_argument("--structure",type=str, default=STRUCTURES[0], help="Song structure")
    parser.add_argument("--tempo",    type=str, default=TEMPO_FEELS[1], help="Tempo feel")
    parser.add_argument("--model",    type=str, default="gemma4", help="Ollama model name (default: gemma4)")
    parser.add_argument("--temp",     type=float, default=0.85, help="Temperature 0.0–1.0 (default: 0.85)")
    parser.add_argument("--output",   type=str, help="Save lyrics to this file path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Determine input mode
    if args.chords and args.genre:
        chords = [c.strip() for c in args.chords.replace(",", " ").split() if c.strip()]
        params = dict(
            chords=chords,
            genre=args.genre,
            theme=args.theme,
            structure=args.structure,
            tempo_feel=args.tempo,
        )
    else:
        params = interactive_mode()

    # Echo parameters
    console.print(Panel(
        f"[bold]Chords:[/bold]   {' → '.join(params['chords'])}\n"
        f"[bold]Genre:[/bold]    {params['genre'].upper()}\n"
        f"[bold]Theme:[/bold]    {params['theme']}\n"
        f"[bold]Structure:[/bold] {params['structure']}\n"
        f"[bold]Tempo:[/bold]    {params['tempo_feel']}",
        title="[yellow]Song Parameters[/yellow]",
        border_style="yellow",
    ))

    # Confirm Ollama is reachable
    model_name = getattr(args, "model", "gemma4")
    try:
        ollama.show(model_name)
    except ollama.ResponseError:
        console.print(f"\n[red]✗ Model '{model_name}' not found in Ollama.[/red]")
        console.print(f"[dim]Run:  ollama pull {model_name}[/dim]\n")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Could not connect to Ollama: {e}[/red]")
        console.print("[dim]Make sure Ollama is running:  ollama serve[/dim]\n")
        sys.exit(1)

    # Generate
    lyrics = generate_lyrics(
        chords=params["chords"],
        genre=params["genre"],
        theme=params["theme"],
        structure=params["structure"],
        tempo_feel=params["tempo_feel"],
        model=model_name,
        temperature=getattr(args, "temp", 0.85),
    )

    # Optionally save
    output_path = getattr(args, "output", None)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            header = (
                f"Chords: {' → '.join(params['chords'])}\n"
                f"Genre: {params['genre'].upper()}\n"
                f"Theme: {params['theme']}\n"
                f"Structure: {params['structure']}\n"
                f"Tempo: {params['tempo_feel']}\n"
                f"{'─' * 40}\n\n"
            )
            f.write(header + lyrics)
        console.print(f"\n[green]✓ Lyrics saved to:[/green] {output_path}")

    console.print("\n[bold green]✓ Done![/bold green]")


if __name__ == "__main__":
    main()