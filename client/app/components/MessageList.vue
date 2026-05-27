<script setup lang="ts">
import type { ChatMessage } from '~/types'

defineProps<{
  messages: readonly ChatMessage[]
  currentStage: string
  currentStageStatus: string
}>()

const listEl = ref<HTMLElement | null>(null)

// Auto-scroll to bottom whenever messages change or text streams in
watch(
  () => listEl.value?.scrollHeight,
  () => {
    nextTick(() => {
      if (listEl.value) {
        listEl.value.scrollTop = listEl.value.scrollHeight
      }
    })
  },
)
</script>

<template>
  <div ref="listEl" class="flex flex-col gap-6 overflow-y-auto px-4 py-6">
    <!-- Empty state -->
    <div v-if="messages.length === 0" class="flex flex-1 flex-col items-center justify-center gap-3 py-24 text-center">
      <div class="text-4xl">🎵</div>
      <h2 class="text-xl font-semibold text-zinc-200">ChordAI</h2>
      <p class="max-w-sm text-sm text-zinc-500">
        Describe a song and I'll generate chord progressions and lyrics.<br>
        Try <em>"something melancholic like early Radiohead"</em> or <em>"upbeat 80s pop song about summer"</em>.
      </p>
    </div>

    <template v-for="msg in messages" :key="msg.id">
      <!-- User message -->
      <div v-if="msg.role === 'user'" class="flex justify-end">
        <div class="max-w-lg rounded-2xl rounded-tr-sm bg-emerald-700 px-4 py-3 text-sm text-white shadow">
          {{ msg.text }}
        </div>
      </div>

      <!-- Assistant message -->
      <div v-else class="flex justify-start">
        <div class="w-full max-w-2xl space-y-3 rounded-2xl rounded-tl-sm bg-zinc-800 px-4 py-4 shadow">
          <!-- Pipeline status while generating -->
          <StatusBar
            v-if="msg.stageStatus !== 'done' && msg.stageStatus !== 'error'"
            :stage="currentStage"
            :status="currentStageStatus"
          />

          <!-- Error state -->
          <div v-if="msg.error" class="rounded-lg bg-red-900/30 px-3 py-2 text-sm text-red-400">
            {{ msg.error }}
          </div>

          <!-- Streaming / done: show SongOutput if result exists, else raw streaming text -->
          <SongOutput v-if="msg.result" :result="msg.result" />
          <pre
            v-else-if="msg.text"
            class="whitespace-pre-wrap font-mono text-sm text-zinc-300 leading-relaxed"
          >{{ msg.text }}<span v-if="msg.stageStatus === 'streaming'" class="animate-pulse">▍</span></pre>

          <!-- Thinking spinner before any text arrives -->
          <div
            v-else-if="!msg.error"
            class="flex items-center gap-2 text-sm text-zinc-500"
          >
            <span class="inline-block h-2 w-2 animate-bounce rounded-full bg-zinc-500" />
            <span class="inline-block h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:150ms]" />
            <span class="inline-block h-2 w-2 animate-bounce rounded-full bg-zinc-500 [animation-delay:300ms]" />
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
