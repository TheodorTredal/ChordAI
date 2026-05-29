// chord_runner calls sample.py from chord-gen/nanoGPT to generate a chord
// token string using the trained GPT checkpoint (out-CHORDv2).
// Falls back to a hardcoded stub if the script or checkpoint is missing.
package models

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"chordai/server/schemas"
)

var (
	reGenre   = regexp.MustCompile(`<genre:([^>]+)>`)
	reDecade  = regexp.MustCompile(`<decade:(\d+)>`)
	reSection = regexp.MustCompile(`^<(\w+)>$`)
)

var knownSections = map[string]bool{
	"intro": true, "verse": true, "prechorus": true, "chorus": true,
	"bridge": true, "outro": true, "solo": true, "interlude": true,
	"instrumental": true, "interlude2": true,
}

// chordCheckpoint is the out-dir name (relative to nanoGPT/) for the model to load.
const chordCheckpoint = "out-CHORDv2"

// genreMap translates planner genre names to the chord model's trained vocabulary.
// Unmapped genres fall back to "pop".
var genreMap = map[string]string{
	"pop":         "pop",
	"rock":        "rock",
	"blues":       "rock",
	"folk":        "alternative",
	"jazz":        "jazz",
	"country":     "country",
	"rnb":         "soul",
	"soul":        "soul",
	"metal":       "metal",
	"hiphop":      "rap",
	"hip-hop":     "rap",
	"rap":         "rap",
	"edm":         "electronic",
	"electronic":  "electronic",
	"punk":        "punk",
	"reggae":      "reggae",
	"alternative": "alternative",
	"pop_rock":    "pop_rock",
}

// genreChords is used only by the stub fallback.
var genreChords = map[string][4]string{
	"pop":     {"C", "G", "Am", "F"},
	"rock":    {"E", "A", "D", "B"},
	"blues":   {"A", "D", "E", "A"},
	"folk":    {"G", "C", "D", "Em"},
	"jazz":    {"Cmaj7", "Am7", "Dm7", "G7"},
	"country": {"G", "C", "D", "Em"},
	"rnb":     {"Am", "F", "C", "G"},
	"metal":   {"Em", "C", "G", "D"},
	"hiphop":  {"Am", "G", "F", "E"},
	"edm":     {"Am", "F", "C", "G"},
}

// RunChordModel generates a SongSpec by running sample.py in chord-gen/nanoGPT.
// Falls back to the stub when the script or checkpoint directory is absent.
func RunChordModel(decision *schemas.PlannerDecision, scriptDir string) (*schemas.SongSpec, error) {
	nanoGPTDir := filepath.Join(scriptDir, "chord-gen", "nanoGPT")
	samplePy := filepath.Join(nanoGPTDir, "sample.py")
	checkpointDir := filepath.Join(nanoGPTDir, chordCheckpoint)

	if _, err := os.Stat(samplePy); os.IsNotExist(err) {
		return runChordStub(decision)
	}
	if _, err := os.Stat(checkpointDir); os.IsNotExist(err) {
		return runChordStub(decision)
	}

	tokenStr, err := callSamplePy(decision, nanoGPTDir)
	if err != nil {
		return nil, err
	}
	return parseTokenString(tokenStr, decision.TempoBPM, decision.Vibe), nil
}

func callSamplePy(decision *schemas.PlannerDecision, nanoGPTDir string) (string, error) {
	mode := decision.Mode
	if mode == "" {
		mode = "generate"
	}

	args := []string{
		"sample.py",
		fmt.Sprintf("--out_dir=%s", chordCheckpoint),
		fmt.Sprintf("--mode=%s", mode),
		fmt.Sprintf("--genre=%s", mapGenre(decision.Genre)),
		fmt.Sprintf("--decade=%d", clampDecade(decision.Decade)),
		"--num_samples=1",
		fmt.Sprintf("--device=%s", detectDevice()),
	}
	if decision.SeedChords != "" {
		args = append(args, fmt.Sprintf("--seed_chords=%s", decision.SeedChords))
	}
	if decision.NextSection != "" {
		args = append(args, fmt.Sprintf("--next_section=%s", decision.NextSection))
	}

	cmd := exec.Command("python3", args...)
	cmd.Dir = nanoGPTDir
	cmd.Stderr = os.Stderr

	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("sample.py exited %d", exitErr.ExitCode())
		}
		return "", fmt.Errorf("chord model subprocess: %w", err)
	}

	tokenStr := extractTokenString(string(out))
	if tokenStr == "" {
		return "", fmt.Errorf("no chord token string in sample.py output")
	}
	return tokenStr, nil
}

func mapGenre(genre string) string {
	if mapped, ok := genreMap[strings.ToLower(strings.TrimSpace(genre))]; ok {
		return mapped
	}
	return "pop"
}

func clampDecade(decade int) int {
	d := (decade / 10) * 10
	if d < 1950 {
		return 1950
	}
	if d > 2020 {
		return 2020
	}
	return d
}

func detectDevice() string {
	if err := exec.Command("nvidia-smi").Run(); err == nil {
		return "cuda"
	}
	return "cpu"
}

// extractTokenString parses sample.py stdout to find the generated token string.
// sample.py prints "=== sample 1 ===" then the token string on the next non-empty line,
// with <bos> stripped and <eos> replaced by "[end]".
func extractTokenString(stdout string) string {
	inSample := false
	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "=== sample") {
			inSample = true
			continue
		}
		if inSample && line != "" {
			// Strip "[end]" marker substituted for <eos> by sample.py
			line = strings.TrimSpace(strings.ReplaceAll(line, "[end]", ""))
			// Restore <bos> so parseTokenString gets a consistent format
			return "<bos> " + line
		}
	}
	return ""
}

// runChordStub returns a plausible progression without calling the Python model.
func runChordStub(decision *schemas.PlannerDecision) (*schemas.SongSpec, error) {
	chords, ok := genreChords[strings.ToLower(decision.Genre)]
	if !ok {
		chords = genreChords["pop"]
	}
	c := chords
	tokenStr := fmt.Sprintf(
		"<bos> <genre:%s> <decade:%d> <verse> %s %s %s %s <chorus> %s %s %s %s <bridge> %s %s %s %s <eos>",
		strings.ToLower(decision.Genre), decision.Decade,
		c[0], c[1], c[2], c[3],
		c[0], c[1], c[2], c[3],
		c[2], c[3], c[0], c[1],
	)
	return parseTokenString(tokenStr, decision.TempoBPM, decision.Vibe), nil
}

// parseTokenString converts a chord token string into a SongSpec.
func parseTokenString(tokenStr string, tempoBPM int, vibe string) *schemas.SongSpec {
	tokens := strings.Fields(tokenStr)

	genre := "pop"
	decade := 2010
	sections := make(map[string][]string)
	var sectionOrder []string
	currentSection := ""

	for _, tok := range tokens {
		if tok == "<bos>" || tok == "<eos>" {
			continue
		}
		if m := reGenre.FindStringSubmatch(tok); m != nil {
			genre = strings.ToLower(m[1])
			continue
		}
		if m := reDecade.FindStringSubmatch(tok); m != nil {
			fmt.Sscanf(m[1], "%d", &decade)
			continue
		}
		if m := reSection.FindStringSubmatch(tok); m != nil && knownSections[m[1]] {
			currentSection = m[1]
			if _, exists := sections[currentSection]; !exists {
				sections[currentSection] = []string{}
				sectionOrder = append(sectionOrder, currentSection)
			}
			continue
		}
		if currentSection != "" && tok != "" {
			sections[currentSection] = append(sections[currentSection], tok)
		}
	}

	_ = sectionOrder

	return &schemas.SongSpec{
		Genre:     genre,
		Decade:    decade,
		TempoBPM:  tempoBPM,
		Vibe:      vibe,
		Sections:  sections,
		RawTokens: tokenStr,
	}
}
