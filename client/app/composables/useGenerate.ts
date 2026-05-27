import type { ChatMessage, PlannerInput, SongResult, WSEvent } from '~/types'

export function useGenerate() {
  const config = useRuntimeConfig()
  const serverUrl = config.public.serverUrl as string

  const messages = ref<ChatMessage[]>([])
  const isGenerating = ref(false)
  const currentStage = ref<string>('')
  const currentStageStatus = ref<string>('idle')

  function addUserMessage(text: string): string {
    const id = crypto.randomUUID()
    messages.value.push({ id, role: 'user', text })
    return id
  }

  function addAssistantMessage(): string {
    const id = crypto.randomUUID()
    messages.value.push({
      id,
      role: 'assistant',
      text: '',
      stageStatus: 'running',
    })
    return id
  }

  function getAssistant(id: string): ChatMessage | undefined {
    return messages.value.find(m => m.id === id)
  }

  async function generate(input: PlannerInput) {
    if (isGenerating.value) return

    isGenerating.value = true
    currentStage.value = ''
    currentStageStatus.value = 'idle'

    addUserMessage(input.freetext || JSON.stringify(input))
    const assistantId = addAssistantMessage()

    try {
      await streamViaWebSocket(input, assistantId)
    } catch (wsErr) {
      // WebSocket failed — fall back to blocking REST call
      console.warn('[useGenerate] WebSocket failed, falling back to REST:', wsErr)
      try {
        await fetchViaRest(input, assistantId)
      } catch (restErr) {
        const msg = getAssistant(assistantId)
        if (msg) {
          msg.error = String(restErr)
          msg.stageStatus = 'error'
        }
      }
    } finally {
      isGenerating.value = false
      currentStage.value = ''
      currentStageStatus.value = 'idle'
    }
  }

  function streamViaWebSocket(input: PlannerInput, assistantId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = serverUrl.replace(/^http/, 'ws') + '/ws/generate'
      const ws = new WebSocket(wsUrl)

      const timeout = setTimeout(() => {
        ws.close()
        reject(new Error('WebSocket connection timeout'))
      }, 10_000)

      ws.onopen = () => {
        clearTimeout(timeout)
        ws.send(JSON.stringify(input))
      }

      ws.onerror = (e) => {
        clearTimeout(timeout)
        reject(new Error('WebSocket error'))
      }

      ws.onmessage = (event) => {
        let wsEvent: WSEvent
        try {
          wsEvent = JSON.parse(event.data)
        } catch {
          return
        }

        handleWSEvent(wsEvent, assistantId, ws, resolve, reject)
      }

      ws.onclose = (e) => {
        if (!e.wasClean) {
          reject(new Error(`WebSocket closed unexpectedly (code ${e.code})`))
        }
      }
    })
  }

  function handleWSEvent(
    event: WSEvent,
    assistantId: string,
    ws: WebSocket,
    resolve: () => void,
    reject: (e: Error) => void,
  ) {
    const msg = getAssistant(assistantId)
    if (!msg) return

    currentStage.value = event.stage
    currentStageStatus.value = event.status

    if (event.status === 'error') {
      msg.error = event.error
      msg.stageStatus = 'error'
      ws.close()
      reject(new Error(event.error))
      return
    }

    if (event.stage === 'lyrics_model' && event.status === 'streaming' && event.token) {
      msg.text += event.token
      msg.stageStatus = 'streaming'
      return
    }

    if (event.stage === 'result' && event.status === 'done' && event.token) {
      try {
        const result: SongResult = JSON.parse(event.token)
        msg.result = result
        msg.text = result.lyrics
        msg.stageStatus = 'done'
      } catch {
        msg.stageStatus = 'done'
      }
      ws.close()
      resolve()
      return
    }

    // Generic stage status update
    msg.stageStatus = event.status
  }

  async function fetchViaRest(input: PlannerInput, assistantId: string) {
    const msg = getAssistant(assistantId)
    if (msg) {
      msg.text = ''
      msg.stageStatus = 'running'
    }

    const resp = await fetch(`${serverUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    })

    if (!resp.ok) {
      const err = await resp.text()
      throw new Error(`Server error ${resp.status}: ${err}`)
    }

    const result: SongResult = await resp.json()

    if (msg) {
      msg.result = result
      msg.text = result.lyrics
      msg.stageStatus = 'done'
    }
  }

  return {
    messages: readonly(messages),
    isGenerating: readonly(isGenerating),
    currentStage: readonly(currentStage),
    currentStageStatus: readonly(currentStageStatus),
    generate,
  }
}
