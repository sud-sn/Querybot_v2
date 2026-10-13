<script setup>
// A member of a tile, right-clicked (static/js/qb-charts.js sends 'qb-menu'): filter the dashboard to it, or break it
// down by another field, a level further on that tile. Where it was clicked; closed by Escape, a press outside it or
// a choice. The arrow keys move through it.
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { icon, t } from '../components/i18n.js'

const props = defineProps({ ask: { type: Object, default: null } })
const emit = defineEmits(['filter', 'drill', 'close'])
const root = ref(null)
const place = ref({ left: '0px', top: '0px' })

function away(event) {
  if (root.value && !root.value.contains(event.target)) emit('close')
}
function key(event) {
  if (event.key === 'Escape') {
    event.stopPropagation()
    emit('close')
    return
  }
  const items = [...(root.value?.querySelectorAll('.tile-menu-item') || [])]
  const at = items.indexOf(document.activeElement)
  const step = { ArrowDown: 1, ArrowUp: -1 }[event.key]
  if (!step || !items.length) return
  event.preventDefault()
  items[(at + step + items.length) % items.length].focus()
}
const stop = () => {
  document.removeEventListener('pointerdown', away, true)
  document.removeEventListener('keydown', key, true)
}
watch(() => props.ask, (ask) => {
  stop()
  if (!ask) return
  document.addEventListener('pointerdown', away, true)
  document.addEventListener('keydown', key, true)
  place.value = { left: `${ask.x}px`, top: `${ask.y}px` }
  nextTick(() => {
    // Kept on the screen: a click near its right or bottom edge opens it to the left of, or above, the pointer.
    const box = root.value?.getBoundingClientRect()
    if (!box) return
    const left = Math.max(12, Math.min(ask.x, window.innerWidth - box.width - 12))
    const top = ask.y + box.height > window.innerHeight - 12 ? Math.max(12, ask.y - box.height) : ask.y
    place.value = { left: `${left}px`, top: `${top}px` }
    root.value?.querySelector('.tile-menu-item')?.focus({ preventScroll: true })
  })
})
onBeforeUnmount(stop)
</script>

<template>
  <Teleport to="body">
    <div v-if="ask" ref="root" class="tile-menu" role="menu" :aria-label="ask.member" :style="place">
      <div class="tile-menu-group">
        <div class="tile-menu-head">{{ ask.member }}</div>
        <button type="button" class="tile-menu-item" role="menuitem" @click="emit('filter', ask)">
          <span class="tile-menu-icon" v-html="icon('filter', 14)"></span>{{ t('ui.dash.drill.filter', { name: ask.member }) }}
        </button>
      </div>
      <div v-if="ask.offers && ask.offers.length" class="tile-menu-group" role="group"
           :aria-label="t('ui.dash.drill.break', { name: ask.member })">
        <div class="tile-menu-head">{{ t('ui.dash.drill.break', { name: ask.member }) }}</div>
        <button v-for="offer in ask.offers" :key="offer.field" type="button" class="tile-menu-item" role="menuitem"
                @click="emit('drill', ask, offer)">
          <span class="tile-menu-icon" v-html="icon('chevron-right', 14)"></span>{{ offer.label }}
        </button>
      </div>
      <div v-else class="tile-menu-group">
        <p class="tile-menu-note" role="status">
          {{ t(ask.offers ? 'ui.dash.drill.none' : 'ui.dash.drill.checking', { name: ask.member }) }}
        </p>
      </div>
    </div>
  </Teleport>
</template>
