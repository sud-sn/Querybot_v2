<script setup>
// One field's filter ("Region: North +1"): its members to tick, searched as you type. The members of a field a
// tile is grouped by come with the page; those of a field the owner added are read when it is opened, as this
// reader may see them (portal/routes.py dashboard_filter_values_api).
import { computed, nextTick, ref, watch } from 'vue'
import { icon, plural, t } from '../components/i18n.js'
import { usePopover } from '../components/popover.js'

const props = defineProps({
  filter: { type: Object, required: true },
  dashboardId: { type: Number, required: true },
  canEdit: { type: Boolean, default: false },
})
const emit = defineEmits(['choose', 'remove'])

const root = ref(null)
const chosen = ref([])
const search = ref('')
const read = ref(null)            // members read from the server, for a field the owner added
const more = ref(false)           // there are more than were listed: the search asks the server
const loading = ref(false)
const problem = ref('')
const open = usePopover(root, { onOpen: start })
const lazy = computed(() => props.filter.values == null)
const selected = computed(() => props.filter.selected || [])
const summary = computed(() => {
  const first = selected.value[0]
  if (!first) return t('ui.dash.all')
  return selected.value.length > 1 ? t('ui.dash.f.and_more', { first, count: selected.value.length - 1 }) : first
})

let asked = 0
async function load(q = '') {
  const ticket = ++asked
  loading.value = true
  problem.value = ''
  try {
    const params = new URLSearchParams({ field: props.filter.field, q })
    const response = await fetch(`/portal/api/dashboard/${props.dashboardId}/filter-values?${params}`,
      { credentials: 'same-origin' })
    const body = await response.json().catch(() => ({}))
    if (ticket !== asked) return
    if (!body.ok) {
      problem.value = body.error || t('ui.dash.f.failed')
      return
    }
    read.value = body.values || []
    if (!q) more.value = Boolean(body.more)
  } catch {
    if (ticket === asked) problem.value = t('ui.dash.f.failed')
  } finally {
    if (ticket === asked) loading.value = false
  }
}
function start() {
  chosen.value = [...selected.value]
  search.value = ''
  if (lazy.value && (read.value == null || more.value)) load()
  nextTick(() => root.value?.querySelector('.mf-search input')?.focus({ preventScroll: true }))
}
let timer = null
watch(search, (q) => {
  if (!lazy.value || !more.value) return
  clearTimeout(timer)
  timer = setTimeout(() => load(q.trim()), 250)
})

// The members on offer, those already ticked first when they are not among them.
const listed = computed(() => {
  const all = lazy.value ? read.value || [] : props.filter.values || []
  const q = search.value.trim().toLowerCase()
  const shown = q ? all.filter((v) => v.toLowerCase().includes(q)) : all
  const kept = chosen.value.filter((v) => !shown.includes(v) && (!q || v.toLowerCase().includes(q)))
  return [...kept, ...shown]
})
function toggle(value) {
  chosen.value = chosen.value.includes(value) ? chosen.value.filter((v) => v !== value) : [...chosen.value, value]
}
function apply() {
  emit('choose', chosen.value)
  open.value = false
}
</script>

<template>
  <div ref="root" class="mf" :class="{ 'is-open': open }">
    <button type="button" class="dash-pick" :class="{ 'is-set': selected.length }" data-trigger aria-haspopup="dialog"
            :aria-expanded="open ? 'true' : 'false'" @click="open = !open">
      <span class="dash-pick-name">{{ filter.label }}</span>
      <span class="dash-pick-value">{{ summary }}</span>
      <span class="dash-pick-caret" v-html="icon('chevron-down', 14)"></span>
    </button>
    <div v-if="open" class="dash-pop mf-panel" role="dialog" :aria-label="filter.label">
      <label class="mf-search">
        <span class="mf-search-icon" v-html="icon('search', 15)"></span>
        <input v-model="search" class="form-input form-input-sm" type="search" :placeholder="t('ui.dash.f.search')"
               :aria-label="t('ui.dash.f.search')">
      </label>
      <div class="mf-list" role="group" :aria-label="filter.label">
        <label v-for="value in listed" :key="value" class="mf-option">
          <input type="checkbox" :checked="chosen.includes(value)" @change="toggle(value)">
          <span>{{ value }}</span>
        </label>
        <p v-if="loading" class="mf-note" role="status">{{ t('ui.dash.f.loading') }}</p>
        <p v-else-if="problem" class="mf-note is-problem" role="alert">{{ problem }}</p>
        <p v-else-if="!listed.length" class="mf-note">{{ t('ui.dash.f.none_found') }}</p>
        <p v-if="more && !search.trim() && !loading" class="mf-note">{{ t('ui.dash.f.more') }}</p>
      </div>
      <div class="mf-foot">
        <button v-if="canEdit && filter.source === 'owner'" type="button" class="btn btn-link btn-sm mf-remove dash-edit-only"
                @click="emit('remove'); open = false">{{ t('ui.dash.f.remove') }}</button>
        <span class="mf-count">{{ chosen.length ? plural('ui.dash.f.chosen', chosen.length) : t('ui.dash.all') }}</span>
        <button type="button" class="btn btn-ghost btn-sm" :disabled="!chosen.length" @click="chosen = []">{{ t('ui.dash.clear') }}</button>
        <button type="button" class="btn btn-primary btn-sm" @click="apply">{{ t('ui.dash.dp.apply') }}</button>
      </div>
    </div>
  </div>
</template>
