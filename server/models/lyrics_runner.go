// lyrics_runner wraps lyric_generator_gemma4.py as a subprocess and reads
// the resulting JSON for clean lyrics output.
package models

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"chordai/server/schemas"
)

// RunLyricsModel calls lyric_generator_gemma4.py, waits for it to finish,
// then reads the lyrics from the saved JSON file.
func RunLyricsModel(spec *schemas.SongSpec, scriptDir string) (string, error) {
	// Script lives at models/lyric_generator_gemma4.py relative to the project root.
	args := []string{
		"models/lyric_generator_gemma4.py",
		"--out_dir", filepath.Join(scriptDir, "out-chords"),
		"--mode", "generate",
		"--genre", spec.Genre,
		"--decade", fmt.Sprintf("%d", spec.Decade),
		"--tempo", fmt.Sprintf("%d", spec.TempoBPM),
		"--vibe", spec.Vibe,
		"--token_string", spec.RawTokens, // passed as separate arg — safe with spaces
	}

	cmd := exec.Command("python3", args...)
	cmd.Dir = scriptDir
	// Route all Python stdout/stderr to the server's stderr so rich console
	// output never leaks into the lyrics result.
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("lyrics model start: %w", err)
	}

	if err := cmd.Wait(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("lyrics model exited %d", exitErr.ExitCode())
		}
		return "", fmt.Errorf("lyrics model: %w", err)
	}

	// Read lyrics from the saved JSON — the single source of truth.
	lyrics, err := readLatestLyricsJSON(filepath.Join(scriptDir, "out-chords"))
	if err != nil {
		return "", fmt.Errorf("lyrics model: could not read output JSON: %w", err)
	}
	return lyrics, nil
}

// readLatestLyricsJSON finds the most recently written lyrics JSON in outDir
// and returns the "lyrics" field as the clean lyrics string.
func readLatestLyricsJSON(outDir string) (string, error) {
	entries, err := os.ReadDir(outDir)
	if err != nil {
		return "", err
	}

	type jsonFile struct {
		name    string
		modTime time.Time
	}
	var files []jsonFile
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "lyrics_") && strings.HasSuffix(e.Name(), ".json") {
			info, err := e.Info()
			if err != nil {
				continue
			}
			files = append(files, jsonFile{name: e.Name(), modTime: info.ModTime()})
		}
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no lyrics JSON found in %s", outDir)
	}

	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.After(files[j].modTime)
	})

	data, err := os.ReadFile(filepath.Join(outDir, files[0].name))
	if err != nil {
		return "", err
	}

	var result struct {
		Lyrics string `json:"lyrics"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", err
	}
	return result.Lyrics, nil
}
