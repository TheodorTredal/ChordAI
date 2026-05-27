package main

import (
	"log"
	"os"
	"path/filepath"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

	"chordai/server/routers"
)

func main() {
	// scriptDir is the project root — one level up from server/
	// so that subprocess calls to sample_chords.py and gemma4_lyrics.py resolve correctly.
	exe, err := os.Executable()
	if err != nil {
		log.Fatalf("cannot resolve executable path: %v", err)
	}
	// During `go run`, the executable lives in a temp dir; fall back to cwd.
	scriptDir := filepath.Join(filepath.Dir(exe), "..")
	if _, err := os.Stat(filepath.Join(scriptDir, "sample_chords.py")); os.IsNotExist(err) {
		// Fallback: assume we're run from the server/ directory
		cwd, _ := os.Getwd()
		scriptDir = filepath.Join(cwd, "..")
	}
	scriptDir = filepath.Clean(scriptDir)
	log.Printf("[main] script dir: %s", scriptDir)

	r := gin.Default()

	// CORS — allow the Nuxt dev server and any client origin during development.
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000", "http://localhost:5173"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		AllowWebSockets:  true,
	}))

	// POST /api/generate — blocking REST endpoint
	api := r.Group("/api")
	routers.RegisterAPIRoutes(api, scriptDir)

	// GET /ws/generate — streaming WebSocket endpoint
	ws := r.Group("/ws")
	routers.RegisterWSRoutes(ws, scriptDir)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("[main] ChordAI server listening on :%s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
