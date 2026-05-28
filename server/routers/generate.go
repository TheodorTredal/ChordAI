// generate.go exposes two endpoints:
//   POST /api/generate   — blocking, returns full SongResult as JSON
//   WS   /ws/generate    — streams WSEvent messages until the song is complete
package routers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"chordai/server/executor"
	"chordai/server/schemas"
	"chordai/server/system_connector"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true }, // CORS handled by Gin middleware
}

// RegisterAPIRoutes mounts the REST endpoint under the given group.
// Resulting path: POST {group}/generate
func RegisterAPIRoutes(rg *gin.RouterGroup, scriptDir string) {
	rg.POST("/generate", restGenerate(scriptDir))
}

// RegisterWSRoutes mounts the WebSocket endpoint under the given group.
// Resulting path: GET {group}/generate
func RegisterWSRoutes(rg *gin.RouterGroup, scriptDir string) {
	rg.GET("/generate", wsGenerate(scriptDir))
}

// restGenerate — POST /api/generate
// Blocks until the full song is ready, then returns JSON.
func restGenerate(scriptDir string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var input schemas.PlannerInput
		if err := c.ShouldBindJSON(&input); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		decision, err := system_connector.Plan(input)
		if err != nil {
			log.Printf("[generate] planner error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "planner failed: " + err.Error()})
			return
		}

		result, err := executor.Run(decision, scriptDir, nil)
		if err != nil {
			log.Printf("[generate] executor error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "pipeline failed: " + err.Error()})
			return
		}

		c.JSON(http.StatusOK, result)
	}
}

// wsGenerate — GET /ws/generate (upgraded to WebSocket)
// Expects the client to send a JSON PlannerInput as the first message after
// the handshake. Then streams WSEvent messages until done or error.
func wsGenerate(scriptDir string) gin.HandlerFunc {
	return func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			log.Printf("[ws/generate] upgrade error: %v", err)
			return
		}
		defer conn.Close()

		// Read the initial PlannerInput from the client.
		_, msg, err := conn.ReadMessage()
		if err != nil {
			log.Printf("[ws/generate] read input: %v", err)
			return
		}

		var input schemas.PlannerInput
		if err := json.Unmarshal(msg, &input); err != nil {
			sendWSEvent(conn, schemas.WSEvent{Stage: "input", Status: "error", Error: "invalid JSON: " + err.Error()})
			return
		}

		// Planner stage
		sendWSEvent(conn, schemas.WSEvent{Stage: "planner", Status: "running"})
		decision, err := system_connector.Plan(input)
		if err != nil {
			sendWSEvent(conn, schemas.WSEvent{Stage: "planner", Status: "error", Error: err.Error()})
			return
		}
		sendWSEvent(conn, schemas.WSEvent{Stage: "planner", Status: "done"})

		// Pipeline execution — emit each stage event over the WebSocket.
		onEvent := func(event schemas.WSEvent) error {
			return sendWSEvent(conn, event)
		}

		result, err := executor.Run(decision, scriptDir, onEvent)
		if err != nil {
			sendWSEvent(conn, schemas.WSEvent{Stage: "pipeline", Status: "error", Error: err.Error()})
			return
		}

		// Send final result as a special "result" stage event with the JSON payload.
		resultBytes, _ := json.Marshal(result)
		sendWSEvent(conn, schemas.WSEvent{Stage: "result", Status: "done", Token: string(resultBytes)})
	}
}

func sendWSEvent(conn *websocket.Conn, event schemas.WSEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, data)
}
