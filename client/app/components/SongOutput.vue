<script setup lang="ts">
import type { SongResult } from '~/types'

const props = defineProps<{ result: SongResult }>()

interface ParsedSection {
  label: string   // e.g. "Verse 1"
  chords: string  // e.g. "E A D B"
  body: string    // lyrics lines only
}

const sections = computed<ParsedSection[]>(() => {
  const raw = props.result.lyrics
  const blocks = raw.split(/\n(?=\[)/).filter(b => b.trim())

  return blocks.map(block => {
    const lines = block.split('\n')

    // First line is the section label: [Verse 1]
    const labelLine = lines[0].trim()
    const label = labelLine.replace(/^\[|\]$/g, '')

    // Find and extract a (Chords: ...) line anywhere in the block
    let chords = ''
    const bodyLines = lines.slice(1).filter(line => {
      const m = line.trim().match(/^\(Chords?:\s*(.+?)\)$/i)
      if (m?.[1]) { chords = m[1].trim(); return false }
      return true
    })

    // If no explicit chord line, fall back to the sections map from the result
    if (!chords) {
      const key = label.toLowerCase().replace(/\s+\d+$/, '') // "verse 1" → "verse"
      const fromSpec = props.result.sections[key]
      if (fromSpec?.length) chords = fromSpec.join(' ')
    }

    return { label, chords, body: bodyLines.join('\n').trim() }
  })
})

function sectionColor(label: string): string {
  const key = label.toLowerCase()
  if (key.startsWith('verse'))     return 'bg-blue-900/40 border-blue-700 text-blue-300'
  if (key.startsWith('chorus'))    return 'bg-emerald-900/40 border-emerald-700 text-emerald-300'
  if (key.startsWith('bridge'))    return 'bg-purple-900/40 border-purple-700 text-purple-300'
  if (key.startsWith('pre'))       return 'bg-amber-900/40 border-amber-700 text-amber-300'
  if (key.startsWith('outro'))     return 'bg-zinc-700/40 border-zinc-600 text-zinc-400'
  if (key.startsWith('intro'))     return 'bg-zinc-700/40 border-zinc-600 text-zinc-400'
  if (key.startsWith('solo'))      return 'bg-rose-900/40 border-rose-700 text-rose-300'
  return 'bg-zinc-700/40 border-zinc-600 text-zinc-400'
}
</script>

<template>
  <div class="space-y-5">
    <!-- Song metadata pills -->
    <div class="flex flex-wrap gap-2 text-xs">
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300 font-medium uppercase tracking-wide">{{ result.genre }}</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300">{{ result.decade }}s</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300">{{ result.tempo_bpm }} BPM</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300 italic">{{ result.vibe }}</span>
    </div>

    <!-- Section chord pills -->
    <div v-if="Object.keys(result.sections).length" class="flex flex-wrap gap-2">
      <div
        v-for="(chords, section) in result.sections"
        :key="section"
        class="rounded-lg border px-3 py-1.5 text-xs font-mono"
        :class="sectionColor(String(section))"
      >
        <span class="font-semibold uppercase tracking-widest mr-2">{{ section }}</span>
        <span class="opacity-60">{{ chords.join('  ') }}</span>
      </div>
    </div>

    <!-- Parsed sections -->
    <div v-if="sections.length" class="space-y-6">
      <div v-for="section in sections" :key="section.label" class="space-y-1">
        <!-- Section header: "Verse 1 — E A D B" -->
        <div class="flex items-baseline gap-2 border-l-2 pl-3" :class="sectionColor(section.label)">
          <span class="font-semibold text-base">{{ section.label }}</span>
          <span v-if="section.chords" class="text-sm font-mono opacity-60">{{ section.chords }}</span>
        </div>
        <!-- Lyrics body -->
        <pre class="whitespace-pre-wrap font-sans text-base text-zinc-200 leading-relaxed pl-3">{{ section.body }}</pre>
      </div>
    </div>

    <!-- Fallback: raw lyrics if parsing produced nothing -->
    <pre v-else class="whitespace-pre-wrap font-sans text-base text-zinc-200 leading-relaxed">{{ result.lyrics }}</pre>

    <!-- Album art -->
    <div v-if="result.image_path" class="mt-4">
      <img
        :src="result.image_path"
        :alt="`${result.genre} album cover`"
        class="rounded-xl w-full max-w-sm object-cover shadow-lg shadow-black/50"
      />
    </div>
  </div>
</template>
