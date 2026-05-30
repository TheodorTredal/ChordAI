// executor runs the pipeline array from a PlannerDecision, calling each model
// in order and passing outputs forward. Status events are emitted via
// EventCallback so the WebSocket router can stream progress to the client.
package executor

import (
    "fmt"
    "log"
    "path/filepath"
    "strings"

    "chordai/server/models"
    "chordai/server/schemas"
)

var validStages = map[string]bool{
    "chord_model":  true,
    "lyrics_model": true,
    "image_model":  true,
}

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

    // Validate all stage names before running anything.
    for _, stage := range decision.Pipeline {
        if !validStages[stage] {
            return nil, fmt.Errorf("unknown pipeline stage %q — valid stages: chord_model, lyrics_model, image_model", stage)
        }
    }

    var spec *schemas.SongSpec
    var lyrics string
    var imagePath string

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

            var err error
            lyrics, err = models.RunLyricsModel(spec, scriptDir)
            if err != nil {
                emit(schemas.WSEvent{Stage: stage, Status: "error", Error: err.Error()})
                return nil, fmt.Errorf("lyrics_model: %w", err)
            }
            emit(schemas.WSEvent{Stage: stage, Status: "done"})

        case "image_model":
            emit(schemas.WSEvent{Stage: stage, Status: "running"})
            prompt := fmt.Sprintf(
                "A %s album cover from the %ds, %s vibe, retro aesthetic, digital art, high quality",
                decision.Genre,
                decision.Decade,
                decision.Vibe,
            )
            var err error
            imagePath, err = models.RunImageModel(prompt, scriptDir)
            if err != nil {
                emit(schemas.WSEvent{Stage: stage, Status: "error", Error: err.Error()})
                return nil, fmt.Errorf("image_model: %w", err)
            }
            emit(schemas.WSEvent{Stage: stage, Status: "done"})

        default:
            // unreachable after upfront validation above
            return nil, fmt.Errorf("unexpected pipeline stage %q", stage)
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
        ImagePath: imageURL(imagePath, scriptDir),
    }, nil
}

// imageURL converts an absolute filesystem path produced by image_generator.py
// into a relative URL served by the Go static file handler at /api/out-chords/.
// If the path isn't under scriptDir/out-chords, it is returned unchanged so
// debugging is still possible.
func imageURL(absPath string, scriptDir string) string {
    if absPath == "" {
        return ""
    }
    outDir := filepath.Join(scriptDir, "out-chords")
    rel, err := filepath.Rel(outDir, absPath)
    if err != nil || strings.HasPrefix(rel, "..") {
        return absPath
    }
    return "/api/out-chords/" + filepath.ToSlash(rel)
}