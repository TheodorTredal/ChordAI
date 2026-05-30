// lyrics_runner wraps lyric_generator_gemma4.py as a subprocess.
// The script prints the path of the saved JSON file as its last stdout line;
// this runner reads that path and extracts the lyrics field.
package models

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"chordai/server/schemas"
)

// RunLyricsModel calls lyric_generator_gemma4.py, waits for it to finish,
// then reads the lyrics from the JSON path printed to stdout.
func RunLyricsModel(spec *schemas.SongSpec, scriptDir string) (string, error) {
	args := []string{
		"models/lyric_generator_gemma4.py",
		"--out_dir", filepath.Join(scriptDir, "out-chords"),
		"--mode", "generate",
		"--genre", spec.Genre,
		"--decade", fmt.Sprintf("%d", spec.Decade),
		"--tempo", fmt.Sprintf("%d", spec.TempoBPM),
		"--vibe", spec.Vibe,
		"--token_string", spec.RawTokens,
	}

	var stdout bytes.Buffer
	cmd := exec.Command("python3", args...)
	cmd.Dir = scriptDir
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr // Rich console output goes to server stderr

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("lyrics model exited %d", exitErr.ExitCode())
		}
		return "", fmt.Errorf("lyrics model: %w", err)
	}

	// The script prints the saved JSON path as the last non-empty line.
	jsonPath := lastLine(stdout.String())
	if jsonPath == "" {
		return "", fmt.Errorf("lyrics model: no output path in stdout")
	}

	return readLyricsFromJSON(jsonPath)
}

func readLyricsFromJSON(jsonPath string) (string, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return "", fmt.Errorf("read lyrics JSON %q: %w", jsonPath, err)
	}
	var result struct {
		Lyrics string `json:"lyrics"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("parse lyrics JSON: %w", err)
	}
	if result.Lyrics == "" {
		return "", fmt.Errorf("lyrics JSON at %q has empty lyrics field", jsonPath)
	}
	return result.Lyrics, nil
}

// lastLine returns the last non-empty line from s.
func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return ""
}
