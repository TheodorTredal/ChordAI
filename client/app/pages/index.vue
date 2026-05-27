<script setup lang="ts">
import type { PlannerInput } from '~/types'

const { messages, isGenerating, currentStage, currentStageStatus, generate } = useGenerate()

function handleSubmit(text: string) {
  const input: PlannerInput = { freetext: text }
  generate(input)
}
</script>

<template>
  <div class="flex h-screen flex-col bg-zinc-900 text-zinc-100">
    <!-- Header -->
    <header class="flex items-center gap-3 border-b border-zinc-800 px-6 py-4">
      <span class="text-lg">🎵</span>
      <span class="font-semibold text-zinc-100">ChordAI</span>
      <span class="ml-auto text-xs text-zinc-600">llama3.2 · gemma4 · local</span>
    </header>

    <!-- Message list fills remaining space -->
    <main class="min-h-0 flex-1 overflow-hidden">
      <MessageList
        :messages="messages"
        :current-stage="currentStage"
        :current-stage-status="currentStageStatus"
        class="h-full"
      />
    </main>

    <!-- Input bar pinned to bottom -->
    <footer class="border-t border-zinc-800 px-4 py-4">
      <div class="mx-auto max-w-2xl">
        <ChatInput :disabled="isGenerating" @submit="handleSubmit" />
        <p class="mt-2 text-center text-xs text-zinc-600">
          Shift+Enter for new line · Enter to send
        </p>
      </div>
    </footer>
  </div>
</template>
