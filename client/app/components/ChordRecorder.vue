<script setup lang="ts">
type RecorderState = 'idle' | 'recording' | 'processing' | 'results' | 'error'

const emit = defineEmits<{ useChords: [text: string] }>()

const state = ref<RecorderState>('idle')
const chords = ref<string[]>([])
const errorMsg = ref('')
const duration = ref(0)
const didSend = ref(false)

let mediaRecorder: MediaRecorder | null = null
let chunks: Blob[] = []
let ticker: ReturnType<typeof setInterval> | null = null

const MAX_SECS = 30

async function startRecording() {
  errorMsg.value = ''
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch {
    errorMsg.value = 'Microphone access denied. Allow microphone access and try again.'
    state.value = 'error'
    return
  }

  chunks = []
  mediaRecorder = new MediaRecorder(stream)

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data)
  }

  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop())
    analyzeRecording()
  }

  mediaRecorder.start()
  state.value = 'recording'
  duration.value = 0

  ticker = setInterval(() => {
    duration.value++
    if (duration.value >= MAX_SECS) stopRecording()
  }, 1000)
}

function stopRecording() {
  if (ticker) { clearInterval(ticker); ticker = null }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop()
  state.value = 'processing'
}

async function analyzeRecording() {
  const mimeType = chunks[0]?.type || 'audio/webm'
  const blob = new Blob(chunks, { type: mimeType })
  const ext = mimeType.includes('ogg') ? 'ogg' : 'webm'
  const form = new FormData()
  form.append('audio_file', blob, `recording.${ext}`)

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 30_000)

  try {
    const res = await fetch('/verify-chords', { method: 'POST', body: form, signal: controller.signal })
    clearTimeout(timeout)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    if (data.status === 'success') {
      chords.value = data.temporary_chords ?? []
      state.value = chords.value.length > 0 ? 'results' : 'error'
      if (chords.value.length === 0)
        errorMsg.value = 'No chords detected — try playing more clearly or for a bit longer.'
    } else {
      errorMsg.value = data.message || 'Analysis failed.'
      state.value = 'error'
    }
  } catch (e) {
    clearTimeout(timeout)
    const isTimeout = e instanceof Error && e.name === 'AbortError'
    errorMsg.value = isTimeout
      ? 'Request timed out — check the SVCO server on c6-8.'
      : 'Could not reach SVCO server. Run on c6-8: uvicorn api:app --port 8001 --host 0.0.0.0'
    state.value = 'error'
  }
}

function sendToChat() {
  emit('useChords', chords.value.join(' '))
  didSend.value = true
  setTimeout(() => { didSend.value = false }, 1500)
}

function reset() {
  state.value = 'idle'
  chords.value = []
  errorMsg.value = ''
  duration.value = 0
  didSend.value = false
}

function fmt(s: number) {
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}
</script>

<template>
  <div class="flex h-full flex-col">
    <!-- Panel header -->
    <div class="border-b border-zinc-800 px-4 py-3 flex items-center gap-2 shrink-0">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="h-4 w-4 text-zinc-400">
        <path d="M7 4a3 3 0 0 1 6 0v6a3 3 0 1 1-6 0V4Z" />
        <path d="M5.5 9.643a.75.75 0 0 0-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5h-1.5v-1.546A6.001 6.001 0 0 0 16 10v-.357a.75.75 0 0 0-1.5 0V10a4.5 4.5 0 0 1-9 0v-.357Z" />
      </svg>
      <span class="text-sm font-semibold text-zinc-300">Chord Recorder</span>
    </div>

    <!-- Body -->
    <div class="flex flex-1 flex-col items-center justify-center px-5 py-6 gap-5">

      <!-- IDLE -->
      <template v-if="state === 'idle'">
        <p class="text-xs text-zinc-500 text-center leading-relaxed">
          Play your chords and get them as text to paste into the chat.
        </p>
        <button
          class="flex h-16 w-16 items-center justify-center rounded-full bg-red-600 text-white shadow-lg transition hover:bg-red-500 active:scale-95"
          @click="startRecording"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="h-7 w-7">
            <path d="M7 4a3 3 0 0 1 6 0v6a3 3 0 1 1-6 0V4Z" />
            <path d="M5.5 9.643a.75.75 0 0 0-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5h-1.5v-1.546A6.001 6.001 0 0 0 16 10v-.357a.75.75 0 0 0-1.5 0V10a4.5 4.5 0 0 1-9 0v-.357Z" />
          </svg>
        </button>
        <span class="text-xs text-zinc-600">Click to record</span>
      </template>

      <!-- RECORDING -->
      <template v-else-if="state === 'recording'">
        <div class="flex items-center gap-2">
          <span class="h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse" />
          <span class="text-sm font-mono text-red-400">{{ fmt(duration) }} / {{ fmt(MAX_SECS) }}</span>
        </div>
        <p class="text-xs text-zinc-500 text-center">Play your chords now…</p>
        <button
          class="flex h-14 w-14 items-center justify-center rounded-full bg-zinc-700 text-white shadow transition hover:bg-zinc-600 active:scale-95"
          @click="stopRecording"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="h-6 w-6">
            <rect x="4" y="4" width="12" height="12" rx="2" />
          </svg>
        </button>
        <span class="text-xs text-zinc-600">Click to stop</span>
      </template>

      <!-- PROCESSING -->
      <template v-else-if="state === 'processing'">
        <div class="h-10 w-10 animate-spin rounded-full border-2 border-zinc-600 border-t-emerald-400" />
        <span class="text-sm text-zinc-400">Analyzing chords…</span>
      </template>

      <!-- RESULTS -->
      <template v-else-if="state === 'results'">
        <div class="w-full space-y-4">
          <p class="text-xs text-zinc-500">Detected sequence:</p>
          <div class="flex flex-wrap gap-2">
            <span
              v-for="(chord, i) in chords"
              :key="i"
              class="rounded-md bg-zinc-700 px-3 py-1 text-sm font-mono text-zinc-100 border border-zinc-600"
            >{{ chord }}</span>
          </div>
          <p class="text-xs text-zinc-600 font-mono break-all">{{ chords.join(' ') }}</p>
        </div>
        <div class="w-full flex flex-col gap-2">
          <button
            class="w-full rounded-xl py-2.5 text-sm font-medium transition"
            :class="didSend
              ? 'bg-emerald-700 text-emerald-200 cursor-default'
              : 'bg-emerald-600 text-white hover:bg-emerald-500 active:scale-95'"
            @click="sendToChat"
          >
            {{ didSend ? 'Added to chat ✓' : 'Use in chat' }}
          </button>
          <button
            class="w-full rounded-xl border border-zinc-700 py-2 text-sm text-zinc-400 transition hover:border-zinc-500 hover:text-zinc-200"
            @click="reset"
          >
            Record again
          </button>
        </div>
      </template>

      <!-- ERROR -->
      <template v-else-if="state === 'error'">
        <div class="rounded-xl bg-red-900/20 border border-red-800/40 px-4 py-3 text-sm text-red-400 text-center leading-relaxed">
          {{ errorMsg }}
        </div>
        <button
          class="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-400 transition hover:border-zinc-500 hover:text-zinc-200"
          @click="reset"
        >
          Try again
        </button>
      </template>

    </div>

    <!-- Footer hint -->
    <div class="border-t border-zinc-800 px-4 py-3 shrink-0">
      <p class="text-xs text-zinc-600 text-center leading-relaxed">
        Results may be inconsistent — use "Record again" to retry.
      </p>
    </div>
  </div>
</template>
