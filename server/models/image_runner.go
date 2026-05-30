// image_runner wraps image_generator.py as a subprocess.
// The script prints the absolute path of the saved PNG as its last stdout line;
// this runner captures that and returns it directly.
package models

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// RunImageModel calls image_generator.py and returns the absolute path of the
// generated PNG, as printed by the script on its last stdout line.
func RunImageModel(prompt string, scriptDir string) (string, error) {
	outDir := filepath.Join(scriptDir, "out-chords")

	var stdout bytes.Buffer
	cmd := exec.Command("python3",
		"models/image_generator.py",
		"--out_dir", outDir,
		"--prompt", prompt,
	)
	cmd.Dir = scriptDir
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("image model exited %d", exitErr.ExitCode())
		}
		return "", fmt.Errorf("image model: %w", err)
	}

	imagePath := lastLine(stdout.String())
	if imagePath == "" {
		return "", fmt.Errorf("image model: no output path in stdout")
	}
	return imagePath, nil
}