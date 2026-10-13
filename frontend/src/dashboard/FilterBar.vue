<script setup>
// The dashboard's filter bar: the dates for every tile, a filter for each field its tiles are grouped by or its
// owner added, and Clear. A choice draws the dashboard again in place (refresh.js); the address follows it.
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { t } from '../components/i18n.js'
import AddFilter from './AddFilter.vue'
import ChartEditor from './ChartEditor.vue'
import DatePicker from './DatePicker.vue'
import MemberFilter from './MemberFilter.vue'
import TileMenu from './TileMenu.vue'
import { refreshPage } from './refresh.js'
import { address, copy, drilledInto, drilledUp, picked, splitsUrl, steady } from './view.js'

const props = defineProps({ store: { type: Object, required: true } })
const state = computed(() => props.store.state)
const changed = computed(() => state.value.picker.period !== state.value.picker.default
  || state.value.filters.some((f) => (f.selected || []).length))

async function go(next, { push = true } = {}) {
  props.store.busy = true
  try {
    const fresh = await refreshPage(address(next), { push })
    if (fresh) props.store.state = steady(state.value, fresh)
  } finally {
    props.store.busy = false
  }
}
function change(edit, options) {
  const next = copy(state.value)
  edit(next)
  return go(next, options)
}
const choosePeriod = (choice) => change((s) => Object.assign(s.picker, choice))
const chooseMembers = (field, values) => change((s) => {
  s.filters.find((f) => f.field === field).selected = values
})
const clearAll = () => change((s) => {
  Object.assign(s.picker, { period: s.picker.default, start: '', end: '' })
  for (const f of s.filters) f.selected = []
})
async function remove(filter) {
  props.store.busy = true
  const response = await fetch(`/portal/api/dashboard/${state.value.dashboard_id}/filters`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'remove', field: filter.field }),
  }).catch(() => null)
  const body = response ? await response.json().catch(() => ({})) : {}
  if (!body.ok) {
    props.store.busy = false
    window.qbToast?.notify?.({ body: body.error || t('ui.dash.f.remove_failed'), tone: 'danger', duration: 4000 })
    return
  }
  await change((s) => { s.filters = s.filters.filter((f) => f.field !== filter.field) })
}
// A filter just added: the same view, drawn again with it in the bar.
const added = () => go(copy(state.value), { push: false })

// A member clicked on a tile (static/js/qb-charts.js sends 'qb-pick'): the dashboard filtered to it, as from its
// filter here -- which the bar is given, under the field's own word, when it has none for that field yet.
function pick(event) {
  const asked = event.detail || {}
  if (!asked.field || !asked.member || props.store.busy) return
  change((s) => {
    let filter = s.filters.find((f) => f.field === asked.field)
    if (!filter) {
      filter = { field: asked.field, label: asked.label || asked.field, values: null, source: 'address', selected: [] }
      s.filters.push(filter)
    }
    filter.selected = picked(filter.selected, asked.member, asked.add)
  })
}

// A member right-clicked on a tile (static/js/qb-charts.js sends 'qb-menu'): its menu, here. Filtering is as a click
// on it; breaking it down draws that tile a level further, the dashboard's other tiles as they were. Of the fields
// the tile offers, the menu lists only those that split this member here (asked of the server as it opens, as the
// chat offers only what an answer can open): never a level of one bar. Each answer is kept for the view it was for.
const menu = ref(null)
const splits = new Map()
let asking = 0
async function openMenu(event) {
  const detail = event.detail
  if (props.store.busy || !detail) {
    menu.value = null
    return
  }
  const ticket = ++asking
  const url = splitsUrl(state.value.dashboard_id, { ...detail, all: detail.offers || [] }, window.location.search)
  menu.value = { ...detail, all: detail.offers || [], offers: splits.has(url) ? splits.get(url) : null, ticket }
  if (splits.has(url)) return
  let kept = detail.offers || []        // when it cannot be asked, the menu offers what the tile does, as before
  try {
    const response = await fetch(url, { credentials: 'same-origin' })
    const body = await response.json().catch(() => ({}))
    if (body.ok) {
      const fields = new Set(body.offers || [])
      kept = kept.filter((offer) => fields.has(offer.field))
      splits.set(url, kept)
    }
  } catch {
    // kept as offered
  }
  if (menu.value && menu.value.ticket === ticket) menu.value.offers = kept
}
function menuFilter(ask) {
  menu.value = null
  pick({ detail: { ...ask, add: false } })
}
function menuDrill(ask, offer) {
  menu.value = null
  change((s) => {
    s.drills = { ...(s.drills || {}), [ask.chart]: drilledInto((s.drills || {})[ask.chart], ask, offer.field) }
  })
}
// The way back up a drilled tile (its path above it, portal_dashboard.html): a step chosen keeps the levels before it.
function drillUp(event) {
  const step = event.target.closest('[data-drill-keep]')
  if (!step || props.store.busy) return
  const tile = step.dataset.drillTile
  change((s) => {
    const drills = { ...(s.drills || {}) }
    const up = drilledUp(drills[tile], Number(step.dataset.drillKeep))
    if (up) drills[tile] = up
    else delete drills[tile]
    s.drills = drills
  })
}

// Back and Forward go through the views chosen here.
const back = () => {
  props.store.busy = true
  refreshPage(window.location.href, { push: false })
    .then((fresh) => { if (fresh) props.store.state = steady(state.value, fresh) })
    .finally(() => { props.store.busy = false })
}

// A tile's editor (ChartEditor.vue), opened from its settings button in Edit; what it saves is drawn again in place.
const editing = ref(null)
function openEditor(event) {
  const button = event.target.closest('[data-edit-tile]')
  if (!button || props.store.busy) return
  const card = button.closest('.chart-card')
  let payload = null
  try {
    payload = JSON.parse(card?.querySelector('[data-chart]')?.dataset.chart || 'null')
  } catch {
    payload = null
  }
  if (!payload) return
  editing.value = {
    id: Number(button.dataset.editTile), payload, palette: button.dataset.editPalette || 'default',
    title: (card.querySelector('.chart-card-title')?.textContent || '').trim(),
    limit: button.dataset.editLimit || '', order: button.dataset.editOrder || '',
  }
}
function edited() {
  editing.value = null
  back()
}
onMounted(() => {
  window.addEventListener('popstate', back)
  window.addEventListener('qb-pick', pick)
  window.addEventListener('qb-menu', openMenu)
  document.addEventListener('click', drillUp)
  document.addEventListener('click', openEditor)
})
onBeforeUnmount(() => {
  window.removeEventListener('popstate', back)
  window.removeEventListener('qb-pick', pick)
  window.removeEventListener('qb-menu', openMenu)
  document.removeEventListener('click', drillUp)
  document.removeEventListener('click', openEditor)
})
</script>

<template>
  <div class="dash-filters" role="group" :aria-label="t('ui.dash.filters_label')" :aria-busy="store.busy ? 'true' : 'false'">
    <DatePicker :picker="state.picker" :dates="state.dates" @choose="choosePeriod" />
    <MemberFilter v-for="filter in state.filters" :key="filter.field" :filter="filter" :dashboard-id="state.dashboard_id"
                  :can-edit="state.can_edit" @choose="(values) => chooseMembers(filter.field, values)" @remove="remove(filter)" />
    <AddFilter v-if="state.can_edit && state.owner_filters < state.most_filters" :dashboard-id="state.dashboard_id" @added="added" />
    <button v-if="changed" type="button" class="dash-filters-clear" @click="clearAll">{{ t('ui.dash.clear') }}</button>
    <span class="dash-filters-busy" :class="{ 'is-on': store.busy }" role="status" aria-live="polite">{{ store.busy ? t('ui.dash.f.updating') : '' }}</span>
    <TileMenu :ask="menu" @filter="menuFilter" @drill="menuDrill" @close="menu = null" />
    <ChartEditor :tile="editing" :dashboard-id="state.dashboard_id" @close="editing = null" @saved="edited" />
  </div>
</template>
