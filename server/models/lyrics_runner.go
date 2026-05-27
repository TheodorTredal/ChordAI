// lyrics_runner wraps gemma4_lyrics.py as a subprocess and streams its output
// token by token via a callback so the executor can forward tokens to the
// WebSocket client in real time.
package models

import (
	"bufio"
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

// TokenCallback is called for each token streamed from the lyrics model.
// Returning an error cancels the stream.
type TokenCallback func(token string) error

// RunLyricsModel calls gemma4_lyrics.py with the given SongSpec and decision,
// streams stdout tokens via onToken, and returns the complete lyrics string.
// If onToken is nil, streaming is skipped and the full output is returned only.
func RunLyricsModel(spec *schemas.SongSpec, scriptDir string, onToken TokenCallback) (string, error) {
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

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("lyrics model stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("lyrics model start: %w", err)
	}

	var sb strings.Builder
	scanner := bufio.NewScanner(stdout)
	scanner.Split(bufio.ScanRunes) // scan character by character for real-time streaming

	for scanner.Scan() {
		token := scanner.Text()
		sb.WriteString(token)
		if onToken != nil {
			if err := onToken(token); err != nil {
				cmd.Process.Kill()
				return sb.String(), err
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return sb.String(), fmt.Errorf("lyrics model exited %d", exitErr.ExitCode())
		}
		return sb.String(), fmt.Errorf("lyrics model: %w", err)
	}

	// If the model saved a JSON file, read lyrics from it as the authoritative source.
	// This handles cases where the model output contains rich/console markup.
	if jsonLyrics, err := readLatestLyricsJSON(filepath.Join(scriptDir, "out-chords")); err == nil {
		return jsonLyrics, nil
	}

	return sb.String(), nil
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
