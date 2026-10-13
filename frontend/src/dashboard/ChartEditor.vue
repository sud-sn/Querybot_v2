<script setup>
// A tile's editor, opened from its settings button in Edit (portal_dashboard.html, data-edit-tile): its name, the
// kinds of chart its rows can be drawn as, its colours, and -- for a chart of members -- how many it shows and in
// what order. The preview draws the tile's own rows as they would look; Save keeps it as a version of the dashboard
// (/portal/api/update-chart) and the dashboard is drawn again in place. Escape, Cancel or a press on the backdrop
// close it unchanged.
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { t } from '../components/i18n.js'
import { arranged } from './view.js'

const props = defineProps({ tile: { type: Object, default: null }, dashboardId: { type: Number, default: 0 } })
const emit = defineEmits(['close', 'saved'])

// Every kind a tile may be drawn as, in the order the page offers them (portal_dashboard.html DASH_TYPES).
const KINDS = ['dumbbell', 'bar', 'stacked', 'line', 'area', 'pie', 'donut', 'scatter', 'heatmap', 'map']
const PALETTES = ['default', 'ocean', 'candy']
const LIMITS = ['5', '10', '20', '']
const ORDERS = ['high', 'low', 'name']

const name = ref('')
const kind = ref('')
const palette = ref('default')
const limit = ref('')
const order = ref('high')
const saving = ref(false)
const problem = ref('')
const panel = ref(null)
const preview = ref(null)
let drawn = null

const kinds = computed(() => (props.tile && window.QBCharts ? window.QBCharts.offeredTypes(props.tile.payload, KINDS) : []))
const shaped = computed(() => Boolean(props.tile && props.tile.order))
const reshaped = computed(() => shaped.value && (limit.value !== props.tile.limit || order.value !== props.tile.order))
const swatches = (set) => (window.QBCharts ? window.QBCharts.paletteFor(set).slice(0, 5) : [])

function draw() {
  if (!props.tile || !preview.value || !window.QBCharts) return
  const payload = {
    ...props.tile.payload,
    chart_type: kind.value || props.tile.payload.chart_type,
    color_palette: palette.value,
    ...(shaped.value ? arranged(props.tile.payload, order.value, limit.value) : {}),
  }
  drawn?.dispose?.()
  drawn = window.QBCharts.render(preview.value, payload, { grow: false })
}

function key(event) {
  if (event.key === 'Escape') {
    event.stopPropagation()
    emit('close')
  }
}

watch(() => props.tile, (tile) => {
  document.removeEventListener('keydown', key, true)
  drawn?.dispose?.()
  drawn = null
  problem.value = ''
  if (!tile) return
  name.value = tile.title
  kind.value = tile.payload.chart_type || ''
  palette.value = PALETTES.includes(tile.palette) ? tile.palette : 'default'
  limit.value = tile.limit
  order.value = tile.order || 'high'
  document.addEventListener('keydown', key, true)
  nextTick(() => {
    draw()
    panel.value?.querySelector('input')?.focus()
  })
})
watch([kind, palette, limit, order], () => nextTick(draw))
onBeforeUnmount(() => {
  document.removeEventListener('keydown', key, true)
  drawn?.dispose?.()
})

async function save() {
  const tile = props.tile
  const changes = {}
  if (name.value.trim() && name.value.trim() !== tile.title) changes.title = name.value.trim()
  if (kind.value && kind.value !== tile.payload.chart_type) changes.chart_type = kind.value
  if (palette.value !== tile.palette) changes.color_palette = palette.value
  if (shaped.value && limit.value !== tile.limit) changes.limit = limit.value || null
  if (shaped.value && order.value !== tile.order) changes.order = order.value
  if (!Object.keys(changes).length) {
    emit('close')
    return
  }
  saving.value = true
  problem.value = ''
  try {
    const response = await fetch('/portal/api/update-chart', {
      method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dashboard_id: props.dashboardId, chart_id: tile.id, ...changes }),
    })
    const body = await response.json().catch(() => ({}))
    if (!response.ok || !body.ok) {
      problem.value = body.error || t('ui.dash.editor.failed')
      return
    }
    emit('saved')
  } catch {
    problem.value = t('ui.dash.editor.failed')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <Teleport to="body">
    <div v-if="tile" class="chart-editor-backdrop" @pointerdown.self="emit('close')">
      <section ref="panel" class="chart-editor" role="dialog" aria-modal="true" :aria-label="t('ui.dash.edit_chart')">
        <header class="chart-editor-head">
          <h2>{{ t('ui.dash.edit_chart') }}</h2>
          <button type="button" class="chart-editor-x" :aria-label="t('ui.dash.editor.cancel')" @click="emit('close')">×</button>
        </header>
        <div class="chart-editor-body">
          <label class="chart-editor-field">
            <span>{{ t('ui.dash.editor.name') }}</span>
            <input v-model="name" class="form-input" type="text" maxlength="120" @keydown.enter.prevent="save">
          </label>
          <fieldset v-if="kinds.length > 1" class="chart-editor-field">
            <legend>{{ t('ui.dash.editor.kind') }}</legend>
            <div class="chart-editor-seg" role="radiogroup">
              <button v-for="k in kinds" :key="k" type="button" role="radio" :aria-checked="k === kind"
                      :class="{ 'is-on': k === kind }" @click="kind = k">{{ t('ui.enum.charttype.' + k) }}</button>
            </div>
          </fieldset>
          <fieldset class="chart-editor-field">
            <legend>{{ t('ui.dash.editor.colours') }}</legend>
            <div class="chart-editor-palettes" role="radiogroup">
              <button v-for="set in PALETTES" :key="set" type="button" role="radio" :aria-checked="set === palette"
                      :class="{ 'is-on': set === palette }" @click="palette = set">
                <span class="chart-editor-swatches"><i v-for="c in swatches(set)" :key="c" :style="{ background: c }"></i></span>
                {{ t('ui.settings.palette.' + set) }}
              </button>
            </div>
          </fieldset>
          <template v-if="shaped">
            <fieldset class="chart-editor-field">
              <legend>{{ t('ui.dash.editor.show') }}</legend>
              <div class="chart-editor-seg" role="radiogroup">
                <button v-for="n in LIMITS" :key="n || 'all'" type="button" role="radio" :aria-checked="n === limit"
                        :class="{ 'is-on': n === limit }" @click="limit = n">
                  {{ n ? t('ui.dash.editor.show_top', { n }) : t('ui.dash.editor.show_all') }}</button>
              </div>
            </fieldset>
            <fieldset class="chart-editor-field">
              <legend>{{ t('ui.dash.editor.order') }}</legend>
              <div class="chart-editor-seg" role="radiogroup">
                <button v-for="o in ORDERS" :key="o" type="button" role="radio" :aria-checked="o === order"
                        :class="{ 'is-on': o === order }" @click="order = o">{{ t('ui.dash.editor.order.' + o) }}</button>
              </div>
            </fieldset>
          </template>
          <div class="chart-editor-field">
            <span>{{ t('ui.dash.editor.preview') }}</span>
            <div ref="preview" class="chart-editor-preview"></div>
            <p v-if="reshaped" class="chart-editor-note">{{ t('ui.dash.editor.redrawn') }}</p>
          </div>
          <p v-if="problem" class="chart-editor-problem" role="alert">{{ problem }}</p>
        </div>
        <footer class="chart-editor-foot">
          <button type="button" class="btn btn-ghost" @click="emit('close')">{{ t('ui.dash.editor.cancel') }}</button>
          <button type="button" class="btn btn-primary" :disabled="saving" @click="save">{{ t('ui.dash.editor.save') }}</button>
        </footer>
      </section>
    </div>
  </Teleport>
</template>
