export interface PlannerInput {
  freetext: string
  genre?: string | null
  decade?: number | null
  tempo_bpm?: number | null
  vibe?: string | null
  mode?: string | null
  seed_chords?: string
  next_section?: string
}

export interface SongResult {
  genre: string
  decade: number
  tempo_bpm: number
  vibe: string
  sections: Record<string, readonly string[]>
  raw_tokens: string
  lyrics: string
  image_path?: string
}

export type PipelineStage = 'planner' | 'chord_model' | 'lyrics_model' | 'image_model' | 'result' | 'pipeline' | 'input'
export type StageStatus = 'idle' | 'running' | 'streaming' | 'done' | 'error'

export interface WSEvent {
  stage: PipelineStage
  status: StageStatus
  token?: string
  error?: string
}

// A single turn in the chat — either from the user or the assistant (song result)
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string          // user prompt text, or streaming lyrics for assistant
  result?: SongResult   // populated once the full result arrives
  stage?: PipelineStage
  stageStatus?: StageStatus
  error?: string
}
