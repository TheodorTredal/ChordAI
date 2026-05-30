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
    var imagePath string // <-- Ny lokal variabel for å holde på stien til det genererte bildet

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

        // --- HER ER DEN NYE STAGE-EN FOR BILDEGENERERING ---
        case "image_model":
            emit(schemas.WSEvent{Stage: stage, Status: "running"})

            // Vi bygger en dynamisk prompt basert på hva Llama-planleggeren fant ut om sangen.
            // Du kan justere denne strengen akkurat slik du vil for å få best stil!
            prompt := fmt.Sprintf(
                "A nostalgic %s album cover from the %ds, %s vibe, retro aesthetic, digital art, high quality", 
                decision.Genre, 
                decision.Decade, 
                decision.Vibe,
            )

            var err2 error
            // Vi kaller funksjonen fra den nye models/image_runner.go filen
            imagePath, err2 = models.RunImageModel(prompt, scriptDir)
            if err2 != nil {
                emit(schemas.WSEvent{Stage: stage, Status: "error", Error: err2.Error()})
                return nil, fmt.Errorf("image_model: %w", err2)
            }
            emit(schemas.WSEvent{Stage: stage, Status: "done"})

        default:
            log.Printf("[executor] unknown pipeline stage %q — skipping", stage)
        }
    }

    if spec == nil {
        return nil, fmt.Errorf("pipeline produced no SongSpec — was chord_model included?")
    }

    // Vi returnerer det fulle resultatet der det nye bildet er bakt inn
    return &schemas.SongResult{
        Genre:     spec.Genre,
        Decade:    spec.Decade,
        TempoBPM:  spec.TempoBPM,
        Vibe:      spec.Vibe,
        Sections:  spec.Sections,
        RawTokens: spec.RawTokens,
        Lyrics:    lyrics,
        ImagePath: imagePath, // <-- Sørg for at dette feltet er lagt til i schemas.SongResult
    }, nil
}