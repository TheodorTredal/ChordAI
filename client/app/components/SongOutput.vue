<script setup lang="ts">
import type { SongResult } from '~/types'

defineProps<{ result: SongResult }>()

function sectionColor(section: string): string {
  const map: Record<string, string> = {
    verse:      'bg-blue-900/40 text-blue-300 border-blue-700',
    chorus:     'bg-emerald-900/40 text-emerald-300 border-emerald-700',
    bridge:     'bg-purple-900/40 text-purple-300 border-purple-700',
    intro:      'bg-zinc-700/40 text-zinc-300 border-zinc-600',
    outro:      'bg-zinc-700/40 text-zinc-300 border-zinc-600',
    prechorus:  'bg-amber-900/40 text-amber-300 border-amber-700',
    solo:       'bg-rose-900/40 text-rose-300 border-rose-700',
    interlude:  'bg-indigo-900/40 text-indigo-300 border-indigo-700',
  }
  return map[section.toLowerCase()] ?? 'bg-zinc-700/40 text-zinc-300 border-zinc-600'
}
</script>

<template>
  <div class="space-y-4">
    <!-- Song metadata pill row -->
    <div class="flex flex-wrap gap-2 text-xs">
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300 font-medium uppercase tracking-wide">{{ result.genre }}</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300">{{ result.decade }}s</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300">{{ result.tempo_bpm }} BPM</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300 italic">{{ result.vibe }}</span>
    </div>

    <!-- Per-section chord + lyrics display -->
    <div v-if="Object.keys(result.sections).length" class="space-y-3">
      <div
        v-for="(chords, section) in result.sections"
        :key="section"
        class="rounded-lg border px-3 py-2 text-xs font-mono"
        :class="sectionColor(String(section))"
      >
        <span class="font-semibold uppercase tracking-widest mr-2">{{ section }}</span>
        <span class="opacity-70">{{ chords.join('  ') }}</span>
      </div>
    </div>

    <!-- Full lyrics -->
    <pre class="whitespace-pre-wrap font-mono text-sm text-zinc-200 leading-relaxed">{{ result.lyrics }}</pre>
  </div>
</template>
