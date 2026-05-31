// chord_runner calls sample.py from chord-gen/nanoGPT to generate chord
// progressions. It runs the model numChordRuns times, compacts each section's
// chord sequence to its base repeating cycle, and merges the results into a
// ChordIdeas palette for the lyrics model.
// Falls back to a hardcoded stub if the script or checkpoint is missing.
package models

import (
	"fmt"
	"log"
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
	"instrumental": true,
}

// chordCheckpoint is the out-dir name (relative to nanoGPT/) for the model to load.
const chordCheckpoint = "out-CHORDv2"

// numChordRuns is how many times sample.py is called to build the chord palette.
const numChordRuns = 3

// genreMap translates planner genre names to the chord model's trained vocabulary.
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

// RunChordModel runs sample.py numChordRuns times, compacts each section's chord
// sequence to its base cycle, and merges the results into a ChordIdeas palette.
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

	var specs []*schemas.SongSpec
	for i := 0; i < numChordRuns; i++ {
		tokenStr, err := callSamplePy(decision, nanoGPTDir)
		if err != nil {
			log.Printf("[chord_runner] run %d/%d failed: %v", i+1, numChordRuns, err)
			if i == 0 {
				return nil, err
			}
			break
		}
		spec := parseTokenString(tokenStr, decision.TempoBPM, decision.Vibe)
		// Compact each section's chord list to its base repeating cycle.
		for section, chords := range spec.Sections {
			spec.Sections[section] = detectCycle(chords)
		}
		specs = append(specs, spec)
	}

	return mergeSpecs(specs), nil
}

// detectCycle reduces a repeated chord sequence to its base cycle.
// e.g. [Am F C G Am F C G Am F C G] → [Am F C G]
//
// If no clean cycle is found and n > 4, it retries on the first 4 chords.
// This catches near-cycles produced by the chord model where the base pattern
// is slightly decorated at the end, e.g. [F G F G F C] → first 4 = [F G F G]
// → period 2 → [F G]. Most musical progressions are 2 or 4 chords, so this
// gives a clean, singable loop in practice.
func detectCycle(chords []string) []string {
	n := len(chords)
	// Try each period length that divides evenly and produces ≥2 repetitions.
	for period := 2; period <= n/2; period++ {
		if n%period != 0 {
			continue
		}
		base := chords[:period]
		repeated := true
		for i := period; i < n; i++ {
			if chords[i] != base[i%period] {
				repeated = false
				break
			}
		}
		if repeated {
			return base
		}
	}
	// No clean cycle found. If the sequence is long, try with the first 4 chords
	// (one standard bar) before giving up.
	if n > 4 {
		return detectCycle(chords[:4])
	}
	return chords
}

// mergeSpecs combines multiple SongSpec runs into one with a ChordIdeas palette.
// The first spec's metadata (genre, decade, etc.) is used as the base.
func mergeSpecs(specs []*schemas.SongSpec) *schemas.SongSpec {
	if len(specs) == 0 {
		return nil
	}
	base := specs[0]

	// Collect all unique compact progressions per section type across all runs.
	chordIdeas := make(map[string][][]string)
	for _, spec := range specs {
		for section, chords := range spec.Sections {
			if len(chords) == 0 {
				continue
			}
			isDuplicate := false
			for _, existing := range chordIdeas[section] {
				if chordsEqual(existing, chords) {
					isDuplicate = true
					break
				}
			}
			if !isDuplicate {
				chordIdeas[section] = append(chordIdeas[section], chords)
			}
		}
	}
	base.ChordIdeas = chordIdeas

	// Update Sections to hold only the first (representative) idea per section
	// so the frontend chord display and fallback path stay clean.
	for section, ideas := range chordIdeas {
		if len(ideas) > 0 {
			base.Sections[section] = ideas[0]
		}
	}

	return base
}

func chordsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
// sample.py prints "=== sample 1 ===" then the token string on the next non-empty
// line, with <bos> stripped and <eos> replaced by "[end]".
func extractTokenString(stdout string) string {
	inSample := false
	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "=== sample") {
			inSample = true
			continue
		}
		if inSample && line != "" {
			line = strings.TrimSpace(strings.ReplaceAll(line, "[end]", ""))
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
