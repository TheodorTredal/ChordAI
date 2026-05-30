<script setup lang="ts">
defineProps<{
  stage: string
  status: string
}>()

const STAGES = [
  { key: 'planner',     label: 'Planner' },
  { key: 'chord_model', label: 'Chords' },
  { key: 'lyrics_model', label: 'Lyrics' },
  { key: 'image_model', label: 'Image' },
]

const LABELS: Record<string, string> = {
  idle:      '',
  running:   'thinking…',
  streaming: 'writing…',
  done:      'done',
  error:     'error',
}

function stageState(key: string, currentStage: string, currentStatus: string): 'idle' | 'active' | 'done' | 'error' {
  const order = STAGES.map(s => s.key)
  const currentIdx = order.indexOf(currentStage)
  const thisIdx = order.indexOf(key)

  if (currentStatus === 'error' && key === currentStage) return 'error'
  if (thisIdx < currentIdx) return 'done'
  if (key === currentStage) return 'active'
  return 'idle'
}
</script>

<template>
  <div class="flex items-center gap-3 text-xs text-zinc-400">
    <template v-for="(s, i) in STAGES" :key="s.key">
      <div class="flex items-center gap-1.5">
        <!-- Dot indicator -->
        <span
          class="h-2 w-2 rounded-full transition-colors"
          :class="{
            'bg-zinc-600':                    stageState(s.key, stage, status) === 'idle',
            'bg-emerald-400 animate-pulse':   stageState(s.key, stage, status) === 'active',
            'bg-emerald-600':                 stageState(s.key, stage, status) === 'done',
            'bg-red-500':                     stageState(s.key, stage, status) === 'error',
          }"
        />
        <span
          :class="{
            'text-zinc-600':   stageState(s.key, stage, status) === 'idle',
            'text-emerald-400 font-medium': stageState(s.key, stage, status) === 'active',
            'text-zinc-400':   stageState(s.key, stage, status) === 'done',
            'text-red-400':    stageState(s.key, stage, status) === 'error',
          }"
        >
          {{ s.label }}
          <span v-if="s.key === stage && status !== 'done'" class="ml-0.5 opacity-70">{{ LABELS[status] }}</span>
        </span>
      </div>
      <!-- Arrow between stages -->
      <span v-if="i < STAGES.length - 1" class="text-zinc-700">›</span>
    </template>
  </div>
</template>
