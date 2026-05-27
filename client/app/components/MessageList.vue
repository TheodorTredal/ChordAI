<script setup lang="ts">
import type { ChatMessage } from '~/types'

const props = defineProps<{
  messages: readonly ChatMessage[]
  currentStage: string
  currentStageStatus: string
}>()

const bottomEl = ref<HTMLElement | null>(null)

function scrollToBottom() {
  nextTick(() => bottomEl.value?.scrollIntoView({ behavior: 'smooth' }))
}

// Scroll when a new message is added
watch(() => props.messages.length, scrollToBottom)

// Scroll when a response finishes (stageStatus goes to done/error)
watch(
  () => props.messages.map(m => m.stageStatus).join(','),
  scrollToBottom,
)
</script>

<template>
  <div class="flex flex-col gap-8 px-6 py-8">
    <!-- Constrained content column -->
    <div class="mx-auto w-full max-w-3xl space-y-8">

      <!-- Empty state -->
      <div v-if="messages.length === 0" class="flex flex-col items-center justify-center gap-4 py-32 text-center">
        <div class="text-5xl">🎵</div>
        <h2 class="text-2xl font-semibold text-zinc-200">ChordAI</h2>
        <p class="max-w-md text-base text-zinc-500">
          Describe a song and I'll generate chord progressions and lyrics.<br>
          Try <em>"something melancholic like early Radiohead"</em> or <em>"upbeat 80s pop song about summer"</em>.
        </p>
      </div>

      <template v-for="msg in messages" :key="msg.id">
        <!-- User message -->
        <div v-if="msg.role === 'user'" class="flex justify-end">
          <div class="max-w-xl rounded-2xl rounded-tr-sm bg-emerald-700 px-5 py-3.5 text-base text-white shadow">
            {{ msg.text }}
          </div>
        </div>

        <!-- Assistant message -->
        <div v-else class="flex justify-start">
          <div class="w-full space-y-4 rounded-2xl rounded-tl-sm bg-zinc-800 px-6 py-5 shadow">
            <!-- Pipeline status while generating -->
            <StatusBar
              v-if="msg.stageStatus !== 'done' && msg.stageStatus !== 'error'"
              :stage="currentStage"
              :status="currentStageStatus"
            />

            <!-- Error state -->
            <div v-if="msg.error" class="rounded-lg bg-red-900/30 px-4 py-3 text-base text-red-400">
              {{ msg.error }}
            </div>

            <!-- Show result only when fully done -->
            <SongOutput v-if="msg.result" :result="msg.result" />

            <!-- Music equaliser animation while waiting -->
            <div v-else-if="!msg.error" class="flex items-end gap-1 h-8">
              <span class="w-1.5 rounded-sm bg-emerald-500 animate-eq1" />
              <span class="w-1.5 rounded-sm bg-emerald-400 animate-eq2" />
              <span class="w-1.5 rounded-sm bg-emerald-500 animate-eq3" />
              <span class="w-1.5 rounded-sm bg-emerald-400 animate-eq4" />
              <span class="w-1.5 rounded-sm bg-emerald-500 animate-eq5" />
              <span class="ml-3 text-sm text-zinc-500 self-center">Composing…</span>
            </div>
          </div>
        </div>
      </template>

      <!-- Scroll anchor -->
      <div ref="bottomEl" />
    </div>
  </div>
</template>
