// The dashboard drawn again under new dates or filters, without a new page: the server draws it at its new
// address (portal/routes.py _render_dashboard) and each card takes its new content. A chart already drawn keeps
// its ECharts instance and eases into its new rows; the address is kept in the history, so Back undoes a choice
// and the link can be shared. Anything this cannot match up -- tiles added or taken away meanwhile, a sign-in
// page -- is opened as a page instead.

let pending = null

export async function refreshPage(url, { push = true } = {}) {
  pending?.abort()
  const request = new AbortController()
  pending = request
  const page = document.querySelector('.dashboard-page')
  page?.setAttribute('aria-busy', 'true')
  try {
    const response = await fetch(url, { signal: request.signal, credentials: 'same-origin' })
    const html = await response.text()
    if (request.signal.aborted) return null
    const doc = new DOMParser().parseFromString(html, 'text/html')
    const state = readState(doc)
    if (!response.ok || !state || !morph(doc)) {
      window.location.assign(url)
      return null
    }
    if (push) window.history.pushState({ dashboard: true }, '', url)
    else window.history.replaceState(window.history.state, '', url)
    return state
  } catch (error) {
    if (error?.name === 'AbortError') return null
    window.location.assign(url)
    return null
  } finally {
    if (pending === request) {
      pending = null
      page?.removeAttribute('aria-busy')
    }
  }
}

function readState(doc) {
  try {
    return JSON.parse(doc.getElementById('dashFilterState')?.textContent || 'null')
  } catch {
    return null
  }
}

function morph(doc) {
  const cards = [...document.querySelectorAll('.chart-card[data-chart-id]')]
  const fresh = new Map([...doc.querySelectorAll('.chart-card[data-chart-id]')].map((c) => [c.dataset.chartId, c]))
  if (cards.length !== fresh.size || cards.some((card) => !fresh.has(card.dataset.chartId))) return false
  for (const card of cards) morphCard(card, fresh.get(card.dataset.chartId))
  // What the dashboard shows, in three lines: there or not, under the new dates.
  const grid = document.querySelector('.chart-grid')
  document.querySelector('.dash-reading')?.remove()
  const reading = doc.querySelector('.dash-reading')
  if (reading && grid) grid.before(document.adoptNode(reading))
  swap('[data-drawn-at]', doc)
  swap('nav.dashboard-tabs', doc)
  // A schedule saved now keeps what is on screen.
  const dialog = document.getElementById('dashScheduleDialog')
  const next = doc.getElementById('dashScheduleDialog')
  if (dialog && next) dialog.dataset.view = next.dataset.view || '{}'
  swap('#dashSchedFilters', doc)
  window.resizeDashboardCharts?.()
  return true
}

function swap(selector, doc) {
  const here = document.querySelector(selector)
  const there = doc.querySelector(selector)
  if (here && there) here.replaceWith(document.adoptNode(there))
}

function morphCard(card, next) {
  const content = card.querySelector('.chart-card-content')
  const fresh = next.querySelector('.chart-card-content')
  if (!content || !fresh) return
  card.classList.toggle('is-kpi', next.classList.contains('is-kpi'))
  const drawn = content.querySelector('[data-chart]')
  const incoming = fresh.querySelector('[data-chart]')
  const chart = drawn && window.echarts ? window.echarts.getInstanceByDom(drawn) : null
  let payload = null
  if (chart && incoming) {
    try {
      payload = JSON.parse(incoming.dataset.chart || '{}')
    } catch {
      payload = null
    }
  }
  if (chart && payload?.rows?.length) {
    // The chart already drawn stays where it is, and moves to its new rows.
    drawn.dataset.chart = incoming.dataset.chart
    incoming.replaceWith(drawn)
    content.replaceChildren(...[...fresh.childNodes].map((node) => document.adoptNode(node)))
    if (payload.chart_id) window._dashPayloads[payload.chart_id] = payload
    const warnings = window.renderChartWarnings?.(payload)
    if (warnings) drawn.insertAdjacentHTML('afterend', warnings)
    window.QBCharts.update(chart, payload, { grow: false })
  } else {
    chart?.dispose()
    content.replaceChildren(...[...fresh.childNodes].map((node) => document.adoptNode(node)))
    const node = content.querySelector('[data-chart]')
    if (node) {
      try {
        window.initDashboardChart(node)
      } catch (error) {
        console.error('dashboard chart failed to render', error)
        window._showDashChartFailure?.(node)
      }
    }
  }
  const controls = content.querySelector('.dash-ctrl-row[data-chart-id]')
  if (controls) {
    try {
      window.mountDashChartControls(controls)
    } catch (error) {
      console.error(error)
    }
  }
}
