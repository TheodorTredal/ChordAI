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
	// so that subprocess calls to models/*.py and chord-gen/nanoGPT/sample.py resolve correctly.
	exe, err := os.Executable()
	if err != nil {
		log.Fatalf("cannot resolve executable path: %v", err)
	}
	// During `go run`, the executable lives in a temp dir; fall back to cwd.
	scriptDir := filepath.Join(filepath.Dir(exe), "..")
	if _, err := os.Stat(filepath.Join(scriptDir, "models")); os.IsNotExist(err) {
		// Fallback: assume we're run from the server/ directory
		cwd, _ := os.Getwd()
		scriptDir = filepath.Join(cwd, "..")
	}
	scriptDir = filepath.Clean(scriptDir)
	log.Printf("[main] script dir: %s", scriptDir)

	r := gin.Default()

	// CORS — allow the Nuxt dev server, SVCO (localhost), and cluster entry node.
	r.Use(cors.New(cors.Config{
		AllowOrigins:    []string{"http://localhost:3000", "http://localhost:5173", "http://localhost:8000", "http://localhost:5555"},
		AllowMethods:    []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:    []string{"Origin", "Content-Type", "Accept"},
		AllowWebSockets: true,
	}))

	// GET /api/ping — connectivity test
	r.GET("/api/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "message": "ChordAI server is reachable"})
	})

	// POST /api/generate — blocking REST endpoint
	api := r.Group("/api")
	routers.RegisterAPIRoutes(api, scriptDir)

	// GET /ws/generate — streaming WebSocket endpoint
	ws := r.Group("/ws")
	routers.RegisterWSRoutes(ws, scriptDir)

	port := os.Getenv("PORT")
	if port == "" {
		port = "5556"
	}

	log.Printf("[main] ChordAI server listening on :%s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
