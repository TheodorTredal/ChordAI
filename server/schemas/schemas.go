package schemas

// PlannerInput is what the client sends to the server.
type PlannerInput struct {
	Freetext    string  `json:"freetext"`
	Genre       *string `json:"genre"`
	Decade      *int    `json:"decade"`
	TempoBPM    *int    `json:"tempo_bpm"`
	Vibe        *string `json:"vibe"`
	Mode        *string `json:"mode"`         // "generate" | "extend" | "section"
	SeedChords  string  `json:"seed_chords"`
	NextSection string  `json:"next_section"`
}

// PlannerDecision is what the system_connector (llama3.2) returns.
type PlannerDecision struct {
	Genre       string   `json:"genre"`
	Decade      int      `json:"decade"`
	TempoBPM    int      `json:"tempo_bpm"`
	Vibe        string   `json:"vibe"`
	Mode        string   `json:"mode"`
	SeedChords  string   `json:"seed_chords"`
	NextSection string   `json:"next_section"`
	Pipeline    []string `json:"pipeline"` // e.g. ["chord_model", "lyrics_model"]
	Reasoning   string   `json:"reasoning"` // logged server-side only
}

// SongSpec is the shared data contract between chord model and lyrics model.
type SongSpec struct {
	Genre      string                 `json:"genre"`
	Decade     int                    `json:"decade"`
	TempoBPM   int                    `json:"tempo_bpm"`
	Vibe       string                 `json:"vibe"`
	Sections   map[string][]string    `json:"sections"`
	RawTokens  string                 `json:"raw_tokens"`
	// ChordIdeas carries compact chord progressions per section from multiple
	// model runs. Each section maps to a list of candidate progressions so the
	// lyrics model can treat them as creative material rather than a strict spec.
	ChordIdeas map[string][][]string  `json:"chord_ideas,omitempty"`
}

// SongResult is the final output returned to the client.
type SongResult struct {
	Genre     string              `json:"genre"`
	Decade    int                 `json:"decade"`
	TempoBPM  int                 `json:"tempo_bpm"`
	Vibe      string              `json:"vibe"`
	Sections  map[string][]string `json:"sections"`
	RawTokens string              `json:"raw_tokens"`
	Lyrics    string              `json:"lyrics"`
	ImagePath string              `json:"image_path,omitempty"`
}

// WSEvent is a WebSocket status/streaming message sent to the client.
type WSEvent struct {
	Stage  string `json:"stage"`
	Status string `json:"status"`          // "running" | "done" | "streaming" | "error"
	Token  string `json:"token,omitempty"` // only for status=streaming
	Error  string `json:"error,omitempty"` // only for status=error
}
