<script setup>
// The owner adds a field to the filter bar (in Edit): any field whose members may be listed, on the tables they
// may read; those the dashboard's charts already use come first. They name it, and everyone who opens the
// dashboard can pick its members.
import { computed, nextTick, ref } from 'vue'
import { icon, t } from '../components/i18n.js'

const props = defineProps({ dashboardId: { type: Number, required: true } })
const emit = defineEmits(['added'])

const dialog = ref(null)
const fields = ref(null)
const problem = ref('')
const query = ref('')
const picked = ref(null)
const label = ref('')
const saving = ref(false)

async function show() {
  picked.value = null
  query.value = ''
  problem.value = ''
  dialog.value.showModal()
  nextTick(() => dialog.value?.querySelector('.af-search input')?.focus())
  if (fields.value) return
  try {
    const response = await fetch(`/portal/api/dashboard/${props.dashboardId}/filter-fields`, { credentials: 'same-origin' })
    const body = await response.json().catch(() => ({}))
    if (body.ok) fields.value = body.fields || []
    else problem.value = body.error || t('ui.dash.f.fields_failed')
  } catch {
    problem.value = t('ui.dash.f.fields_failed')
  }
}

// By table, those the charts use first.
const groups = computed(() => {
  const q = query.value.trim().toLowerCase()
  const found = (fields.value || []).filter((f) => !q || f.label.toLowerCase().includes(q) || f.table.toLowerCase().includes(q))
  const out = []
  const used = found.filter((f) => f.used)
  if (used.length) out.push({ name: t('ui.dash.f.used'), fields: used })
  const tables = new Map()
  for (const field of found.filter((f) => !f.used)) {
    if (!tables.has(field.table)) tables.set(field.table, [])
    tables.get(field.table).push(field)
  }
  for (const [name, list] of tables) out.push({ name, fields: list })
  return out
})

function pick(field) {
  picked.value = field
  label.value = field.label
  problem.value = ''
  nextTick(() => dialog.value?.querySelector('.af-name input')?.select())
}
async function add() {
  if (!picked.value || !label.value.trim() || saving.value) return
  saving.value = true
  problem.value = ''
  try {
    const response = await fetch(`/portal/api/dashboard/${props.dashboardId}/filters`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add', field: picked.value.field, label: label.value.trim() }),
    })
    const body = await response.json().catch(() => ({}))
    if (!body.ok) {
      problem.value = body.error || t('ui.dash.f.add_failed')
      return
    }
    fields.value = (fields.value || []).filter((f) => f.field !== picked.value.field)
    dialog.value.close()
    window.qbToast?.notify?.({ body: t('ui.dash.f.added', { name: label.value.trim() }), tone: 'success', duration: 3000 })
    emit('added')
  } catch {
    problem.value = t('ui.dash.f.add_failed')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <button type="button" class="dash-pick dash-pick-add dash-edit-only" aria-haspopup="dialog" @click="show">
    <span class="dash-pick-icon" v-html="icon('plus', 14)"></span>
    <span class="dash-pick-value">{{ t('ui.dash.f.add') }}</span>
  </button>
  <dialog ref="dialog" class="dash-dialog af-dialog" aria-labelledby="dashAddFilterTitle">
    <div class="dash-dialog-head">
      <h2 id="dashAddFilterTitle">{{ t('ui.dash.f.add_title') }}</h2>
      <button type="button" class="btn btn-ghost btn-icon btn-sm" :aria-label="t('ui.pin.close')" @click="dialog.close()" v-html="icon('x', 16)"></button>
    </div>
    <p class="af-intro">{{ t('ui.dash.f.add_intro') }}</p>
    <template v-if="!picked">
      <label class="af-search">
        <span class="af-search-icon" v-html="icon('search', 15)"></span>
        <input v-model="query" class="form-input" type="search" :placeholder="t('ui.dash.f.search_fields')" :aria-label="t('ui.dash.f.search_fields')">
      </label>
      <div class="af-list">
        <p v-if="!fields && !problem" class="af-note" role="status">{{ t('ui.dash.f.loading') }}</p>
        <p v-if="problem" class="af-note is-problem" role="alert">{{ problem }}</p>
        <section v-for="group in groups" :key="group.name" class="af-group">
          <h3>{{ group.name }}</h3>
          <button v-for="field in group.fields" :key="field.field" type="button" class="af-field" @click="pick(field)">
            <span class="af-field-name">{{ field.label }}</span>
            <span class="af-field-table">{{ field.table }}</span>
          </button>
        </section>
        <p v-if="fields && !groups.length" class="af-note">{{ t('ui.dash.f.no_fields') }}</p>
      </div>
    </template>
    <form v-else class="af-name" @submit.prevent="add">
      <label class="af-label"><span>{{ t('ui.dash.f.name') }}</span>
        <input v-model="label" class="form-input" maxlength="60" required></label>
      <p class="af-chosen">{{ t('ui.dash.f.field_from', { field: picked.label, table: picked.table }) }}</p>
      <p v-if="problem" class="af-note is-problem" role="alert">{{ problem }}</p>
      <div class="dash-dialog-foot">
        <button type="button" class="btn btn-ghost" @click="picked = null">{{ t('ui.dash.f.back') }}</button>
        <button type="submit" class="btn btn-primary" :disabled="saving || !label.trim()">{{ t('ui.dash.f.add_button') }}</button>
      </div>
    </form>
  </dialog>
</template>
