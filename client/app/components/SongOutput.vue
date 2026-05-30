<script setup lang="ts">
import type { SongResult } from '~/types'

const props = defineProps<{ result: SongResult }>()

// 1. Hent ut runtime config for å få tak i Go-serverens URL (f.eks. http://localhost:8000)
const config = useRuntimeConfig()

// 2. Lag en computed variabel som lager den fulle URL-en til bildet hvis image_path eksisterer.
// Hvis Go-rutene dine ligger under en `/api`-gruppe (f.eks rg := r.Group("/api")), 
// må du legge til `/api/` i strengen under: `${config.public.serverUrl}/api/${props.result.image_path}`
// const albumCoverUrl = computed(() => {
//   if (!props.result.image_path) return null
//   return `${config.public.serverUrl}/${props.result.image_path}`
// })

// const albumCoverUrl = computed(() => {
//   if (!props.result.image_path) return null

//   // Sørger for at vi ikke får doble skråstreker hvis serverUrl slutter på /
//   const baseUrl = config.public.serverUrl.replace(/\/$/, '')

//   // HER definerer du mappen bildet faktisk ligger i på serveren.
//   // Hvis du må gå inn i "api/out-chords", blir det slik:
//   // return `${baseUrl}/ChordAI/out-chords/${props.result.image_path}`
//   // return '/image_1780061810.png'
//   // return `../../out-chords/${props.result.image_path}`
//   // return `../../out-chords/${props.result.image_path}`
//   // return `../../out-chords/`
//   return `/../../out-chords/image_1780165532.png`
// })

const albumCoverUrl = computed(() => {
  if (!props.result.image_path) return null
  
  // Siden backenden sender hele bildet som "data:image/png;base64,...",
  // dytter vi den bare rett inn i <img> taggen!
  return props.result.image_path
})
// ChordAI/out-chords/image_1780061810.png

interface ParsedSection {
  label: string   
  chords: string  
  body: string    
}

const sections = computed<ParsedSection[]>(() => {
  const raw = props.result.lyrics
  if (!raw) return [] // Sikring mot tom tekst
  const blocks = raw.split(/\n(?=\[)/).filter(b => b.trim())

  return blocks.map(block => {
    const lines = block.split('\n')
    const labelLine = lines[0].trim()
    const label = labelLine.replace(/^\[|\]$/g, '')

    let chords = ''
    const bodyLines = lines.slice(1).filter(line => {
      const m = line.trim().match(/^\(Chords?:\s*(.+?)\)$/i)
      if (m?.[1]) { chords = m[1].trim(); return false }
      return true
    })

    if (!chords) {
      const key = label.toLowerCase().replace(/\s+\d+$/, '') 
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
    
    <div v-if="albumCoverUrl" class="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-3 max-w-sm mx-auto md:mx-0">
      <div class="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Generert Albumcover</div>
      <img 
        :src="albumCoverUrl" 
        alt="AI-generert albumcover" 
        class="h-auto w-full rounded-lg object-cover aspect-square border border-zinc-800 shadow-lg"
      />
    </div>

    <div class="flex flex-wrap gap-2 text-xs">
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300 font-medium uppercase tracking-wide">{{ result.genre }}</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300">{{ result.decade }}s</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300">{{ result.tempo_bpm }} BPM</span>
      <span class="rounded-full bg-zinc-700 px-3 py-1 text-zinc-300 italic">{{ result.vibe }}</span>
    </div>

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

    <div v-if="sections.length" class="space-y-6">
      <div v-for="section in sections" :key="section.label" class="space-y-1">
        <div class="flex items-baseline gap-2 border-l-2 pl-3" :class="sectionColor(section.label)">
          <span class="font-semibold text-base">{{ section.label }}</span>
          <span v-if="section.chords" class="text-sm font-mono opacity-60">{{ section.chords }}</span>
        </div>
        <pre class="whitespace-pre-wrap font-sans text-base text-zinc-200 leading-relaxed pl-3">{{ section.body }}</pre>
      </div>
    </div>

    <pre v-else class="whitespace-pre-wrap font-sans text-base text-zinc-200 leading-relaxed">{{ result.lyrics }}</pre>
  </div>
</template>