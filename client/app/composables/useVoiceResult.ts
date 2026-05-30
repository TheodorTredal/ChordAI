import type { ChatMessage, SongResult } from '~/types'
import type { VoiceEvent } from '../../server/voice-bus'

/**
 * Opens a persistent SSE connection to /api/voice-stream.
 *
 * Two event types arrive:
 *   { type: 'start', freetext }   — SVCO began; insert a pending assistant message
 *   { type: 'result', ...SongResult } — pipeline done; fill in the pending message
 *
 * This gives the user immediate feedback (the equaliser animation) the moment
 * SVCO finishes recording, rather than silent waiting for 30-60 seconds.
 */
export function useVoiceResult(messages: Ref<readonly ChatMessage[]>) {
  const writableMessages = messages as Ref<ChatMessage[]>

  onMounted(() => {
    const source = new EventSource('/api/voice-stream')

    // Track the id of the pending assistant message so we can update it in-place
    let pendingId: string | null = null

    source.onmessage = (event) => {
      let evt: VoiceEvent
      try {
        evt = JSON.parse(event.data)
      } catch {
        console.warn('[useVoiceResult] could not parse SSE payload', event.data)
        return
      }

      if (evt.type === 'start') {
        // SVCO just started — show the user message and a loading assistant bubble
        const userText = evt.freetext?.trim() || 'Voice input'

        writableMessages.value.push({
          id: crypto.randomUUID(),
          role: 'user',
          text: `🎙 ${userText}`,
        })

        pendingId = crypto.randomUUID()
        writableMessages.value.push({
          id: pendingId,
          role: 'assistant',
          text: '',
          stageStatus: 'running',
        })
        return
      }

      if (evt.type === 'result') {
        let result: SongResult
        try {
          // evt is { type: 'result', ...SongResult fields }
          const { type: _type, ...rest } = evt as Record<string, unknown>
          result = rest as unknown as SongResult
        } catch {
          console.warn('[useVoiceResult] malformed result payload', evt)
          return
        }

        if (pendingId) {
          // Replace the pending bubble in-place
          const idx = writableMessages.value.findIndex((m: ChatMessage) => m.id === pendingId)
          if (idx !== -1) {
            writableMessages.value[idx] = {
              ...writableMessages.value[idx],
              text: result.lyrics ?? '',
              result,
              stageStatus: 'done',
            }
          }
          pendingId = null
        } else {
          // No pending bubble (e.g. page reloaded mid-session) — just append
          writableMessages.value.push({
            id: crypto.randomUUID(),
            role: 'assistant',
            text: result.lyrics ?? '',
            result,
            stageStatus: 'done',
          })
        }
      }
    }

    source.onerror = () => {
      // SSE auto-reconnects; if there was a pending bubble, mark it errored
      if (pendingId) {
        const idx = writableMessages.value.findIndex((m: ChatMessage) => m.id === pendingId)
        if (idx !== -1) {
          writableMessages.value[idx] = {
            ...writableMessages.value[idx],
            stageStatus: 'error',
            error: 'Voice connection lost',
          }
        }
        pendingId = null
      }
    }

    onUnmounted(() => source.close())
  })
}
