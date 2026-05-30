// system_connector calls llama3.2 via the Ollama REST API to interpret user
// input and produce a validated PlannerDecision. It uses Ollama's structured
// output (format field with JSON Schema) so the model is constrained to always
// produce valid JSON — no retry loop needed.
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
	ollamaURL    = "http://localhost:11434/api/chat"
	plannerModel = "llama3.2"
	temperature  = 0.3
)

// plannerSchema is the JSON Schema passed to Ollama's `format` field.
// Ollama uses constrained decoding to guarantee the output matches this schema.
// Must stay in sync with schemas.PlannerDecision and its json tags.
var plannerSchema = json.RawMessage(`{
  "type": "object",
  "required": ["genre","decade","tempo_bpm","vibe","mode","seed_chords","next_section","pipeline","reasoning"],
  "properties": {
    "genre":        { "type": "string" },
    "decade":       { "type": "integer", "minimum": 1950, "maximum": 2020 },
    "tempo_bpm":    { "type": "integer", "minimum": 40,   "maximum": 200  },
    "vibe":         { "type": "string" },
    "mode":         { "type": "string", "enum": ["generate","extend","section"] },
    "seed_chords":  { "type": "string" },
    "next_section": { "type": "string" },
    "pipeline":     { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "reasoning":    { "type": "string" }
  },
  "additionalProperties": false
}`)

// ollamaChatRequest is the payload sent to the Ollama /api/chat endpoint.
type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Format   json.RawMessage `json:"format"`
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
You are an AI music planner. Interpret the user's song request and fill in every field of the required JSON schema.

Rules:
- Use any explicit values the user provides (genre, decade, etc.) as-is.
- Infer missing values from the freetext description.
- "mode" must be "generate" unless seed_chords are provided (then "extend") or a next_section is specified (then "section").
- CRITICAL: If "Seed chords (played by user)" are provided, copy them EXACTLY as-is into the "seed_chords" field. Do not rephrase, reorder, or drop any chord. The chord model requires the exact sequence the user played.
- "pipeline" must always include "chord_model". Add "lyrics_model" unless the user explicitly asks for chords only. Add "image_model" only if the user asks for an image or album cover.
- "tempo_bpm" must be between 40 and 200.
- "decade" must be a multiple of 10, clamped to 1950–2020.
- "reasoning" is for your internal notes and will not be shown to the user.
`)

// Plan interprets a PlannerInput using llama3.2 and returns a PlannerDecision.
// Ollama's constrained JSON output guarantees valid JSON — no retry needed.
func Plan(input schemas.PlannerInput) (*schemas.PlannerDecision, error) {
	userMsg := buildUserMessage(input)

	reqBody := ollamaChatRequest{
		Model: plannerModel,
		Messages: []ollamaMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userMsg},
		},
		Stream:  false,
		Format:  plannerSchema,
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

	var decision schemas.PlannerDecision
	if err := json.Unmarshal([]byte(ollamaResp.Message.Content), &decision); err != nil {
		return nil, fmt.Errorf("parse planner JSON: %w", err)
	}

	log.Printf("[system_connector] pipeline=%v reasoning=%s", decision.Pipeline, decision.Reasoning)
	return &decision, nil
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
		fmt.Fprintf(&b, "Seed chords (played by user): %s\n", input.SeedChords)
		fmt.Fprintf(&b, "IMPORTANT: Copy these chords exactly into seed_chords — do not change them.\n")
	}
	if input.NextSection != "" {
		fmt.Fprintf(&b, "Next section: %s\n", input.NextSection)
	}

	return b.String()
}
