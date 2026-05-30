<script setup lang="ts">
import type { PlannerInput } from '~/types'

const { messages, isGenerating, currentStage, currentStageStatus, generate } = useGenerate()

const config = useRuntimeConfig()
const pingStatus = ref<'idle' | 'ok' | 'error'>('idle')
const pingMessage = ref('')

const chatInput = ref<{ append(text: string): void } | null>(null)

function handleSubmit(text: string) {
  const input: PlannerInput = { freetext: text }
  generate(input)
}

function handleUseChords(chordText: string) {
  chatInput.value?.append(chordText)
}

async function ping() {
  pingStatus.value = 'idle'
  pingMessage.value = 'Pinging...'
  try {
    const res = await fetch(`${config.public.serverUrl}/api/ping`)
    const data = await res.json()
    pingStatus.value = 'ok'
    pingMessage.value = data.message
  } catch (e) {
    pingStatus.value = 'error'
    pingMessage.value = 'Cannot reach server at ' + config.public.serverUrl
  }
}
</script>

<template>
  <div class="flex h-screen flex-col bg-zinc-900 text-zinc-100">
    <!-- Header -->
    <header class="border-b border-zinc-800 px-6 py-4 shrink-0">
      <div class="mx-auto flex max-w-3xl items-center gap-3">
        <span class="text-xl">🎵</span>
        <span class="text-lg font-semibold text-zinc-100">ChordAI</span>
        <div class="ml-auto flex items-center gap-3">
          <span
            v-if="pingMessage"
            class="text-sm"
            :class="pingStatus === 'ok' ? 'text-emerald-400' : pingStatus === 'error' ? 'text-red-400' : 'text-zinc-500'"
          >{{ pingMessage }}</span>
          <button
            class="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-400 hover:border-zinc-500 hover:text-zinc-200 transition-colors"
            @click="ping"
          >Ping server</button>
        </div>
      </div>
    </header>

    <!-- Two-column body -->
    <div class="flex flex-1 min-h-0">

      <!-- Chat column -->
      <div class="flex flex-col flex-1 min-w-0">
        <main class="min-h-0 flex-1 overflow-y-auto">
          <MessageList
            :messages="messages"
            :current-stage="currentStage"
            :current-stage-status="currentStageStatus"
          />
        </main>
        <footer class="border-t border-zinc-800 px-6 py-6 shrink-0">
          <div class="mx-auto max-w-3xl space-y-2">
            <ChatInput ref="chatInput" :disabled="isGenerating" @submit="handleSubmit" />
            <p class="text-center text-sm text-zinc-600">
              Shift+Enter for new line · Enter to send
            </p>
          </div>
        </footer>
      </div>

      <!-- Chord recorder sidebar -->
      <aside class="w-72 shrink-0 border-l border-zinc-800">
        <ChordRecorder @use-chords="handleUseChords" />
      </aside>

    </div>
  </div>
</template>
