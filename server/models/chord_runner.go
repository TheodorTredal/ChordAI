// chord_runner builds a SongSpec from the PlannerDecision.
// When sample_chords.py is available, swap the stub for a real subprocess call.
package models

import (
	"fmt"
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
}

// genreChords provides a small set of genre-appropriate default progressions
// used by the stub until sample_chords.py is integrated.
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

// RunChordModel generates a SongSpec. Currently uses a stub that produces a
// plausible token string from the PlannerDecision without calling sample_chords.py.
func RunChordModel(decision *schemas.PlannerDecision, scriptDir string) (*schemas.SongSpec, error) {
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

// extractTokenString finds the chord token string in the subprocess stdout.
func extractTokenString(stdout string) string {
	for _, line := range strings.Split(stdout, "\n") {
		if strings.Contains(line, "<bos>") || strings.Contains(line, "<genre:") {
			return strings.TrimSpace(line)
		}
	}
	return ""
}

// parseTokenString converts a chord token string into a SongSpec.
func parseTokenString(tokenStr string, tempoBPM int, vibe string) *schemas.SongSpec {
	tokens := strings.Fields(tokenStr)

	genre := "pop"
	decade := 2010
	sections := make(map[string][]string)
	// preserve insertion order via a slice
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

	_ = sectionOrder // available for ordered iteration if needed downstream

	return &schemas.SongSpec{
		Genre:     genre,
		Decade:    decade,
		TempoBPM:  tempoBPM,
		Vibe:      vibe,
		Sections:  sections,
		RawTokens: tokenStr,
	}
}
