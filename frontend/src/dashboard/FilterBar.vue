<script setup>
// The dashboard's filter bar: the dates for every tile, a filter for each field its tiles are grouped by or its
// owner added, and Clear. A choice draws the dashboard again in place (refresh.js); the address follows it.
import { computed, onBeforeUnmount, onMounted } from 'vue'
import { t } from '../components/i18n.js'
import AddFilter from './AddFilter.vue'
import DatePicker from './DatePicker.vue'
import MemberFilter from './MemberFilter.vue'
import { refreshPage } from './refresh.js'
import { address, copy, picked, steady } from './view.js'

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

// Back and Forward go through the views chosen here.
const back = () => {
  props.store.busy = true
  refreshPage(window.location.href, { push: false })
    .then((fresh) => { if (fresh) props.store.state = steady(state.value, fresh) })
    .finally(() => { props.store.busy = false })
}
onMounted(() => {
  window.addEventListener('popstate', back)
  window.addEventListener('qb-pick', pick)
})
onBeforeUnmount(() => {
  window.removeEventListener('popstate', back)
  window.removeEventListener('qb-pick', pick)
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
  </div>
</template>
