/**
 * In-process bus shared between the two voice API routes.
 *
 * Two event types flow through this bus:
 *   { type: 'start', freetext: string }   — SVCO began processing
 *   { type: 'result', ... SongResult }    — pipeline finished
 *
 * resultQueue holds events that arrived before a browser connected.
 */

export type VoiceEvent =
  | { type: 'start'; freetext: string }
  | { type: 'result'; [key: string]: unknown }

type Listener = (event: VoiceEvent) => void

export const resultQueue: VoiceEvent[] = []
const listeners: Set<Listener> = new Set()

export function addListener(fn: Listener): void {
  listeners.add(fn)
}

export function removeListener(fn: Listener): void {
  listeners.delete(fn)
}

export function notifyListeners(event: VoiceEvent): void {
  resultQueue.push(event)
  for (const fn of listeners) {
    fn(event)
  }
}
