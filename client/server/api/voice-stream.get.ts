/**
 * GET /api/voice-stream
 *
 * SSE endpoint. The browser connects once on page load and keeps the
 * connection open. Both 'start' and 'result' VoiceEvents are forwarded
 * as SSE messages so the frontend can show a loading state immediately
 * when SVCO begins processing, then replace it with the result.
 */

import { resultQueue, addListener, removeListener, type VoiceEvent } from '../voice-bus'

export default defineEventHandler((event) => {
  const res = event.node.res

  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders()

  // Flush any events that arrived before this connection opened
  while (resultQueue.length > 0) {
    const evt = resultQueue.shift()
    res.write(`data: ${JSON.stringify(evt)}\n\n`)
  }

  // Push future events immediately
  const onEvent = (evt: VoiceEvent) => {
    res.write(`data: ${JSON.stringify(evt)}\n\n`)
  }
  addListener(onEvent)

  event.node.req.on('close', () => {
    removeListener(onEvent)
  })

  return new Promise(() => {})
})
