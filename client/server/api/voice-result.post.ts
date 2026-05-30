/**
 * POST /api/voice-result
 *
 * Accepts two payloads from svco/sender.py:
 *   { type: 'start', freetext: string }   — sent immediately when SVCO begins
 *   { type: 'result', ...SongResult }     — sent when the pipeline finishes
 *
 * Both are forwarded to connected browsers via /api/voice-stream (SSE).
 */

import { notifyListeners, type VoiceEvent } from '../voice-bus'

export default defineEventHandler(async (event) => {
  const body = await readBody(event)

  if (!body || typeof body !== 'object') {
    throw createError({ statusCode: 400, message: 'Expected a JSON body' })
  }

  notifyListeners(body as VoiceEvent)

  return { ok: true }
})
