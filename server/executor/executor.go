// executor runs the pipeline array from a PlannerDecision, calling each model
// in order and passing outputs forward. Status events are emitted via
// EventCallback so the WebSocket router can stream progress to the client.
package executor

import (
	"fmt"
	"log"

	"chordai/server/models"
	"chordai/server/schemas"
)

// EventCallback receives WebSocket events as the pipeline runs.
// Returning an error cancels execution.
type EventCallback func(event schemas.WSEvent) error

// Run executes the pipeline described in decision.Pipeline and returns the
// final SongResult. scriptDir is the directory containing the Python scripts.
// onEvent may be nil (used for the non-streaming REST endpoint).
func Run(decision *schemas.PlannerDecision, scriptDir string, onEvent EventCallback) (*schemas.SongResult, error) {
	emit := func(e schemas.WSEvent) {
		if onEvent != nil {
			if err := onEvent(e); err != nil {
				log.Printf("[executor] event callback error: %v", err)
			}
		}
	}

	var spec *schemas.SongSpec
	var lyrics string

	for _, stage := range decision.Pipeline {
		switch stage {
		case "chord_model":
			emit(schemas.WSEvent{Stage: stage, Status: "running"})
			var err error
			spec, err = models.RunChordModel(decision, scriptDir)
			if err != nil {
				emit(schemas.WSEvent{Stage: stage, Status: "error", Error: err.Error()})
				return nil, fmt.Errorf("chord_model: %w", err)
			}
			emit(schemas.WSEvent{Stage: stage, Status: "done"})

		case "lyrics_model":
			if spec == nil {
				return nil, fmt.Errorf("lyrics_model requires chord_model to run first")
			}
			emit(schemas.WSEvent{Stage: stage, Status: "running"})

			tokenCb := func(token string) error {
				emit(schemas.WSEvent{Stage: stage, Status: "streaming", Token: token})
				return nil
			}
			if onEvent == nil {
				tokenCb = nil // no streaming for REST endpoint
			}

			var err error
			lyrics, err = models.RunLyricsModel(spec, scriptDir, tokenCb)
			if err != nil {
				emit(schemas.WSEvent{Stage: stage, Status: "error", Error: err.Error()})
				return nil, fmt.Errorf("lyrics_model: %w", err)
			}
			emit(schemas.WSEvent{Stage: stage, Status: "done"})

		default:
			log.Printf("[executor] unknown pipeline stage %q — skipping", stage)
		}
	}

	if spec == nil {
		return nil, fmt.Errorf("pipeline produced no SongSpec — was chord_model included?")
	}

	return &schemas.SongResult{
		Genre:     spec.Genre,
		Decade:    spec.Decade,
		TempoBPM:  spec.TempoBPM,
		Vibe:      spec.Vibe,
		Sections:  spec.Sections,
		RawTokens: spec.RawTokens,
		Lyrics:    lyrics,
	}, nil
}
