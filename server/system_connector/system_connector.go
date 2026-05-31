// system_connector calls llama3.2 via the Ollama REST API to interpret user
// input and produce a validated PlannerDecision. It is the AI routing layer
// that sits between the client and the downstream chord/lyrics models.
package system_connector

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"chordai/server/schemas"
)

const (
	ollamaURL   = "http://localhost:11434/api/chat"
	plannerModel = "llama3.2"
	temperature  = 0.3
)

// ollamaChatRequest is the payload sent to the Ollama /api/chat endpoint.
type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  ollamaOptions   `json:"options"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature"`
}

type ollamaChatResponse struct {
	Message ollamaMessage `json:"message"`
}

var systemPrompt = strings.TrimSpace(`
You are an AI music planner. Your job is to interpret user input about a song they want to create and produce a structured plan for the downstream chord and lyrics generation models.

You MUST respond with ONLY a valid JSON object matching this exact schema — no preamble, no explanation, no markdown fences:

{
  "genre": "<string: e.g. pop, rock, blues, folk, jazz, country, rnb, metal, hiphop, edm>",
  "decade": <integer: 4-digit year like 1990 or 2010>,
  "tempo_bpm": <integer: beats per minute, e.g. 88>,
  "vibe": "<string: emotional description, e.g. melancholic and detached>",
  "mode": "<string: one of generate | extend | section>",
  "seed_chords": "<string: chord progression if mode is extend or section, else empty string>",
  "next_section": "<string: section name if mode is section, else empty string>",
  "pipeline": ["chord_model", "lyrics_model"],
  "reasoning": "<string: brief explanation of your interpretation>"
}

Rules:
- If the user provides explicit structured values (genre, decade, etc.), use them as-is.
- If any value is missing, infer it from the freetext description.
- "mode" should be "generate" unless seed_chords are provided (then "extend") or a next_section is specified (then "section").
- "pipeline" should always be ["chord_model", "lyrics_model"] unless the user explicitly asks for chords only (then ["chord_model"]) or lyrics only (then ["lyrics_model"]).
- "tempo_bpm" must be an integer between 40 and 200.
- "decade" must be a multiple of 10 between 1950 and 2020.
- "seed_chords" must use the chord model's vocabulary notation. Normalise whatever the user wrote:
    * Sharps use "s" not "#": C# → Cs, F# → Fs, G# → Gs
    * Flats use "b": Bb, Eb, Ab (unchanged)
    * Major chords have no quality suffix: F major → F, Fmaj → F, F:maj → F
    * Minor chords use "min": A minor → Amin, Am → Amin, A:min → Amin
    * Extensions attach directly: Am7 → Amin7, C#m7 → Csmin7, Fmaj7 stays Fmaj7
    * If the user says the chords belong to a specific section (e.g. "these are my chorus chords"), prepend the section tag: <chorus> F G E7 Amin. Valid tags: <verse>, <chorus>, <bridge>, <intro>, <outro>, <solo>, <interlude>.
    * If no section is specified, write the chords without a tag.
- The "reasoning" field is for your internal notes — it will not be shown to the user.
`)

// Plan interprets a PlannerInput using llama3.2 and returns a PlannerDecision.
// It retries once with a correction prompt if JSON parsing fails.
func Plan(input schemas.PlannerInput) (*schemas.PlannerDecision, error) {
	userMsg := buildUserMessage(input)

	decision, err := callOllama(userMsg)
	if err != nil {
		// Retry once with an explicit correction prompt
		log.Printf("[system_connector] first attempt failed (%v), retrying with correction prompt", err)
		correctionMsg := fmt.Sprintf(
			"Your previous response could not be parsed as valid JSON. %v\n\nRespond with ONLY a valid JSON object matching the schema. No preamble.",
			err,
		)
		decision, err = callOllama(correctionMsg)
		if err != nil {
			return nil, fmt.Errorf("system_connector: both attempts failed: %w", err)
		}
	}

	log.Printf("[system_connector] reasoning: %s", decision.Reasoning)
	return decision, nil
}

func buildUserMessage(input schemas.PlannerInput) string {
	var b strings.Builder
	b.WriteString("Plan a song with the following parameters:\n\n")

	if input.Freetext != "" {
		fmt.Fprintf(&b, "Description: %s\n", input.Freetext)
	}
	if input.Genre != nil {
		fmt.Fprintf(&b, "Genre: %s\n", *input.Genre)
	}
	if input.Decade != nil {
		fmt.Fprintf(&b, "Decade: %d\n", *input.Decade)
	}
	if input.TempoBPM != nil {
		fmt.Fprintf(&b, "Tempo BPM: %d\n", *input.TempoBPM)
	}
	if input.Vibe != nil {
		fmt.Fprintf(&b, "Vibe: %s\n", *input.Vibe)
	}
	if input.Mode != nil {
		fmt.Fprintf(&b, "Mode: %s\n", *input.Mode)
	}
	if input.SeedChords != "" {
		fmt.Fprintf(&b, "Seed chords: %s\n", input.SeedChords)
	}
	if input.NextSection != "" {
		fmt.Fprintf(&b, "Next section: %s\n", input.NextSection)
	}

	return b.String()
}

func callOllama(userMsg string) (*schemas.PlannerDecision, error) {
	reqBody := ollamaChatRequest{
		Model: plannerModel,
		Messages: []ollamaMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userMsg},
		},
		Stream:  false,
		Options: ollamaOptions{Temperature: temperature},
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(ollamaURL, "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("ollama unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}

	raw := strings.TrimSpace(ollamaResp.Message.Content)

	// Strip markdown code fences if the model added them despite instructions
	raw = stripCodeFences(raw)

	var decision schemas.PlannerDecision
	if err := json.Unmarshal([]byte(raw), &decision); err != nil {
		return nil, fmt.Errorf("parse JSON from model output (%q): %w", truncate(raw, 200), err)
	}

	return &decision, nil
}

func stripCodeFences(s string) string {
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
