package models

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// RunImageModel kjører image_generator.py som en subprosess,
// og returnerer den absolutte filbanen til det genererte bildet.
func RunImageModel(prompt string, scriptDir string) (string, error) {
	outDir := filepath.Join(scriptDir, "out-chords")

	args := []string{
		// "models/image_generator.py",
		// "image_generator/image_generator.py",
		"models/image_generator.py",
		"--out_dir", outDir,
		"--prompt", prompt,
	}

	cmd := exec.Command("python3", args...)
	cmd.Dir = scriptDir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("image model start: %w", err)
	}

	if err := cmd.Wait(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("image model exited %d", exitErr.ExitCode())
		}
		return "", fmt.Errorf("image model: %w", err)
	}

	// Finn det ferskeste bildet som ble lagret
	imagePath, err := readLatestImagePNG(outDir)
	if err != nil {
		return "", fmt.Errorf("image model: kunne ikke finne ut-fil: %w", err)
	}

	return imagePath, nil
}

// readLatestImagePNG finner den nyeste PNG-filen i ut-mappen
func readLatestImagePNG(outDir string) (string, error) {
	entries, err := os.ReadDir(outDir)
	if err != nil {
		return "", err
	}

	type pngFile struct {
		path    string
		modTime time.Time
	}
	var files []pngFile

	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "image_") && strings.HasSuffix(e.Name(), ".png") {
			info, err := e.Info()
			if err != nil {
				continue
			}
			files = append(files, pngFile{
				path:    filepath.Join(outDir, e.Name()),
				modTime: info.ModTime(),
			})
		}
	}

	if len(files) == 0 {
		return "", fmt.Errorf("ingen bilder funnet i %s", outDir)
	}

	// Sorter etter nyeste først
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.After(files[j].modTime)
	})

	return files[0].path, nil
}