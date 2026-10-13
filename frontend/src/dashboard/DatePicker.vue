<script setup>
// The dates every tile is drawn for: a preset ("This month", the default; "Last quarter"; the fiscal ones where
// the fiscal year does not start in January), two days picked on the calendar, or each tile's own window. Each
// tile counts by its own date; the panel says which ones the tiles use.
import { computed, nextTick, reactive, ref } from 'vue'
import { icon, plural, t } from '../components/i18n.js'
import { usePopover } from '../components/popover.js'
import * as D from './dates.js'

const props = defineProps({
  picker: { type: Object, required: true },
  dates: { type: Array, default: () => [] },
})
const emit = defineEmits(['choose'])

const root = ref(null)
const draft = reactive({ start: '', end: '', hover: '', preview: null, focus: '', month: '' })
const open = usePopover(root, { onOpen: reset })
const lang = computed(() => props.picker.lang || 'en')
const weekdays = computed(() => D.weekdays(lang.value))
const months = computed(() => (draft.month ? [draft.month, D.addMonths(draft.month, 1)] : []))

function reset() {
  draft.start = props.picker.from || ''
  draft.end = props.picker.to || ''
  draft.hover = ''
  draft.preview = null
  draft.focus = draft.end || props.picker.today
  // The month the span ends in on the right, the one before it on the left.
  draft.month = D.addMonths(D.monthStart(draft.focus), -1)
  nextTick(() => {
    // The focus on what is picked: its preset (in sight in its own list, without moving the page), or the
    // calendar's last day of two picked there.
    const on = props.picker.period === 'custom'
      ? root.value?.querySelector(`[data-day="${draft.focus}"]`)
      : root.value?.querySelector('.dp-preset.is-on') || root.value?.querySelector('.dp-preset')
    const list = on?.closest('.dp-presets')
    if (list) list.scrollTop += on.getBoundingClientRect().top - list.getBoundingClientRect().top - 48
    on?.focus({ preventScroll: true })
  })
}

// What the calendar shows: a preset under the pointer, else the days picked so far.
const span = computed(() => {
  if (draft.preview) return draft.preview
  if (draft.start && !draft.end) return draft.hover ? [draft.start, draft.hover].sort() : [draft.start, draft.start]
  return draft.start && draft.end ? [draft.start, draft.end] : null
})
const inSpan = (day) => Boolean(day && span.value && day >= span.value[0] && day <= span.value[1])
const isStart = (day) => Boolean(span.value && day === span.value[0])
const isEnd = (day) => Boolean(span.value && day === span.value[1])
const ready = computed(() => Boolean(draft.start && draft.end && draft.start <= draft.end))
const status = computed(() => {
  if (draft.start && !draft.end) return t('ui.dash.dp.pick_end')
  return ready.value ? D.spanLabel(draft.start, draft.end, lang.value) : ''
})

function choose(period) {
  emit('choose', { period, start: '', end: '' })
  open.value = false
}
// A preset under the pointer, or reached with the keyboard, shows its days on the calendar.
function preview(preset, event) {
  if (event?.type === 'focus' && !event.target.matches(':focus-visible')) return
  draft.preview = preset && preset.start ? [preset.start, preset.end] : null
  if (draft.preview) draft.month = D.addMonths(D.monthStart(preset.end), -1)
}
function pick(day) {
  draft.preview = null
  draft.focus = day
  if (!draft.start || draft.end) {
    draft.start = day
    draft.end = ''
  } else if (day < draft.start) {
    draft.end = draft.start
    draft.start = day
  } else {
    draft.end = day
  }
}
function apply() {
  if (!ready.value) return
  emit('choose', { period: 'custom', start: draft.start, end: draft.end })
  open.value = false
}
function shift(n) {
  draft.month = D.addMonths(draft.month, n)
}
// The arrows move a day or a week, Page Up and Page Down a month; the calendar follows.
function onKey(event) {
  const days = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }[event.key]
  const monthsBy = { PageUp: -1, PageDown: 1 }[event.key]
  if (!days && !monthsBy) return
  event.preventDefault()
  const next = days ? D.addDays(draft.focus, days) : D.addMonths(draft.focus, monthsBy)
  draft.focus = next
  if (next < months.value[0]) draft.month = D.monthStart(next)
  else if (next >= D.addMonths(months.value[1], 1)) draft.month = D.addMonths(D.monthStart(next), -1)
  if (draft.start && !draft.end) draft.hover = next
  nextTick(() => root.value?.querySelector(`[data-day="${next}"]`)?.focus({ preventScroll: true }))
}
</script>

<template>
  <div ref="root" class="dp" :class="{ 'is-open': open }">
    <button type="button" class="dash-pick is-set" data-trigger aria-haspopup="dialog" :aria-expanded="open ? 'true' : 'false'"
            :title="t('ui.dash.dp.title')" @click="open = !open">
      <span class="dash-pick-icon" v-html="icon('calendar', 15)"></span>
      <span class="dash-pick-value">{{ picker.label }}</span>
      <span v-if="picker.span" class="dash-pick-note">{{ picker.span }}</span>
      <span class="dash-pick-caret" v-html="icon('chevron-down', 14)"></span>
    </button>
    <div v-if="open" class="dash-pop dp-panel" role="dialog" :aria-label="t('ui.dash.dp.title')">
      <div class="dp-presets">
        <div v-for="group in picker.groups" :key="group.key" class="dp-group" role="group" :aria-label="group.label">
          <div class="dp-group-name" aria-hidden="true">{{ group.label }}</div>
          <button v-for="preset in group.presets" :key="preset.key" type="button" class="dp-preset"
                  :class="{ 'is-on': picker.period === preset.key }" :aria-pressed="picker.period === preset.key ? 'true' : 'false'"
                  @click="choose(preset.key)" @mouseenter="preview(preset)" @mouseleave="preview(null)"
                  @focus="preview(preset, $event)" @blur="preview(null)">
            <span>{{ preset.label }}</span>
            <span v-if="picker.period === preset.key" class="dp-check" v-html="icon('check', 14)"></span>
          </button>
        </div>
        <div class="dp-group dp-group-own">
          <button type="button" class="dp-preset" :class="{ 'is-on': picker.period === 'own' }"
                  :aria-pressed="picker.period === 'own' ? 'true' : 'false'" :title="t('ui.dash.dp.own_hint')" @click="choose('own')">
            <span>{{ t('ui.dash.period.own') }}</span>
            <span v-if="picker.period === 'own'" class="dp-check" v-html="icon('check', 14)"></span>
          </button>
        </div>
      </div>
      <div class="dp-custom">
        <div class="dp-custom-head">
          <span class="dp-custom-title">{{ t('ui.dash.period.custom') }}</span>
          <span class="dp-nav">
            <button type="button" class="btn btn-ghost btn-icon btn-xs" :aria-label="t('ui.dash.dp.prev')" @click="shift(-1)" v-html="icon('chevron-left', 16)"></button>
            <button type="button" class="btn btn-ghost btn-icon btn-xs" :aria-label="t('ui.dash.dp.next')" @click="shift(1)" v-html="icon('chevron-right', 16)"></button>
          </span>
        </div>
        <div class="dp-months" @mouseleave="draft.hover = ''">
          <div v-for="month in months" :key="month" class="dp-month">
            <div class="dp-month-title">{{ D.monthTitle(month, lang) }}</div>
            <table class="dp-grid" role="grid" :aria-label="D.monthTitle(month, lang)">
              <thead>
                <tr><th v-for="day in weekdays" :key="day.long" scope="col" :abbr="day.long">{{ day.short }}</th></tr>
              </thead>
              <tbody>
                <tr v-for="(week, w) in D.weeks(month)" :key="w">
                  <td v-for="(day, i) in week" :key="i" :class="{ 'in-span': inSpan(day), 'is-start': isStart(day), 'is-end': isEnd(day) }">
                    <button v-if="day" type="button" class="dp-day" :data-day="day" :tabindex="day === draft.focus ? 0 : -1"
                            :class="{ 'is-today': day === picker.today, 'is-edge': isStart(day) || isEnd(day) }"
                            :aria-label="D.dayLabel(day, lang)" :aria-pressed="inSpan(day) ? 'true' : 'false'"
                            @click="pick(day)" @mouseenter="draft.hover = day" @keydown="onKey">{{ Number(day.slice(8)) }}</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="dp-range">
          <label class="dp-field"><span>{{ t('ui.dash.dp.from') }}</span>
            <input v-model="draft.start" class="form-input form-input-sm" type="date" :max="draft.end || undefined" @input="draft.preview = null"></label>
          <span class="dp-dash" aria-hidden="true">–</span>
          <label class="dp-field"><span>{{ t('ui.dash.dp.to') }}</span>
            <input v-model="draft.end" class="form-input form-input-sm" type="date" :min="draft.start || undefined" @input="draft.preview = null"></label>
        </div>
        <p v-if="picker.note" class="dp-dates dp-behind">
          <span class="dp-dates-icon" v-html="icon('clock', 14)"></span><span>{{ picker.note }}</span>
        </p>
        <p v-if="dates.length > 1" class="dp-dates">
          <span class="dp-dates-icon" v-html="icon('info', 14)"></span>
          <span>{{ t('ui.dash.dp.own_dates') }}
            <span v-for="(date, i) in dates" :key="date.label" class="dp-date">{{ date.label }} ({{ plural('ui.dash.dp.charts', date.tiles) }}){{ i < dates.length - 1 ? ', ' : '' }}</span></span>
        </p>
        <p v-else-if="dates.length === 1" class="dp-dates">
          <span class="dp-dates-icon" v-html="icon('info', 14)"></span>
          <span>{{ t('ui.dash.dp.one_date', { date: dates[0].label }) }}</span>
        </p>
        <div class="dp-foot">
          <span class="dp-status" aria-live="polite">{{ status }}</span>
          <button type="button" class="btn btn-ghost btn-sm" @click="open = false">{{ t('ui.dash.dp.cancel') }}</button>
          <button type="button" class="btn btn-primary btn-sm" :disabled="!ready" @click="apply">{{ t('ui.dash.dp.apply') }}</button>
        </div>
      </div>
    </div>
  </div>
</template>
