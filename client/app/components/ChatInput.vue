<script setup lang="ts">
const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ submit: [text: string] }>()

const text = ref('')

function handleSubmit() {
  const trimmed = text.value.trim()
  if (!trimmed || props.disabled) return
  emit('submit', trimmed)
  text.value = ''
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSubmit()
  }
}
</script>

<template>
  <div class="flex items-end gap-4 rounded-2xl border border-zinc-700 bg-zinc-800 px-5 py-4 shadow-lg focus-within:border-zinc-500 transition-colors">
    <textarea
      v-model="text"
      :disabled="disabled"
      rows="3"
      placeholder="Describe the song you want… e.g. 'something melancholic like early Radiohead'"
      class="flex-1 resize-none bg-transparent text-base text-zinc-100 placeholder-zinc-500 outline-none leading-7 max-h-48 overflow-y-auto disabled:opacity-40"
      @keydown="handleKeydown"
      @input="($event.target as HTMLTextAreaElement).style.height = 'auto'; ($event.target as HTMLTextAreaElement).style.height = ($event.target as HTMLTextAreaElement).scrollHeight + 'px'"
    />
    <button
      :disabled="disabled || !text.trim()"
      class="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-500 text-white transition-all hover:bg-emerald-400 disabled:opacity-30 disabled:cursor-not-allowed"
      @click="handleSubmit"
    >
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="h-5 w-5">
        <path d="M3.105 2.288a.75.75 0 0 0-.826.95l1.414 4.926A1.5 1.5 0 0 0 5.135 9.25h6.115a.75.75 0 0 1 0 1.5H5.135a1.5 1.5 0 0 0-1.442 1.086l-1.414 4.926a.75.75 0 0 0 .826.95 28.897 28.897 0 0 0 15.293-7.154.75.75 0 0 0 0-1.115A28.897 28.897 0 0 0 3.105 2.288Z" />
      </svg>
    </button>
  </div>
</template>
