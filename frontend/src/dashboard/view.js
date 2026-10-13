// The address a view is seen at: the dashboard, its tab, the dates picked (none for the default) and each
// filter's members, one f.<field> per member -- as portal/dashboard_view.py reads it.
export function address(state) {
  const params = new URLSearchParams({ dashboard_id: String(state.dashboard_id), tab: state.tab })
  const picker = state.picker
  if (picker.period !== picker.default) params.set('period', picker.period)
  if (picker.period === 'custom') {
    params.set('from', picker.start)
    params.set('to', picker.end)
  }
  for (const filter of state.filters) {
    for (const value of filter.selected || []) params.append(`f.${filter.field}`, value)
  }
  // The tiles drilled into: each member drilled through, in order, and what the deepest level is broken down by.
  for (const [tile, drill] of Object.entries(state.drills || {}).sort(([a], [b]) => Number(a) - Number(b))) {
    for (const step of drill.path || []) params.append(`d.${tile}`, `${step.field}:${step.member}`)
    params.set(`by.${tile}`, drill.by)
  }
  return `/portal/dashboard?${params}`
}

// A tile's drill after a member of it was broken down by a field: one level further. And after a step of its path
// back up was chosen: the levels to keep (none: the tile as pinned), the next level's field what it is broken
// down by.
export function drilledInto(drill, step, by) {
  return { path: [...((drill && drill.path) || []), { field: step.field, member: step.member }], by }
}
export function drilledUp(drill, keep) {
  if (!drill || keep <= 0) return null
  const path = drill.path.slice(0, keep)
  return { path, by: drill.path[keep] ? drill.path[keep].field : drill.by }
}

// A copy of the state to change and ask for.
export const copy = (state) => JSON.parse(JSON.stringify(state))

// The bar drawn again keeps its filters where they were: a field a tile offers can drop out while other filters
// leave that tile one member or none, and the next field would take its place. Those stay, in their order, until
// the page is opened again; a filter its owner took off goes.
export function steady(before, after) {
  if (!before || before.dashboard_id !== after.dashboard_id || before.tab !== after.tab) return after
  const fresh = new Map(after.filters.map((f) => [f.field, f]))
  const filters = []
  for (const old of before.filters) {
    if (fresh.has(old.field)) filters.push(fresh.get(old.field))
    else if (old.source === 'tile') filters.push({ ...old, selected: [] })
    fresh.delete(old.field)
  }
  return { ...after, filters: [...filters, ...fresh.values()] }
}

// The members a click on a tile's member leaves chosen (static/js/qb-charts.js): that member alone, or none when
// it was the only one chosen (the click clears the filter); with Ctrl, Cmd or Shift, the member added to those
// chosen, or taken out of them.
export function picked(now, member, add) {
  const chosen = (now || []).map(String)
  if (add) return chosen.includes(member) ? chosen.filter((m) => m !== member) : [...chosen, member]
  return chosen.length === 1 && chosen[0] === member ? [] : [member]
}

// Where the dashboard is asked what a right-clicked member splits into (portal/routes.py tile_splits_api): the
// page's own address (its dates, filters and drilled tiles), the member, the field it is a member of, and each field
// its menu offers.
export function splitsUrl(dashboardId, ask, search) {
  const params = new URLSearchParams(search || '')
  for (const key of ['member', 'field', 'offer']) params.delete(key)
  params.set('member', ask.member)
  params.set('field', ask.field)
  for (const offer of ask.all || []) params.append('offer', offer.field)
  return `/portal/api/dashboard/${dashboardId}/tiles/${ask.chart}/splits?${params}`
}

// A chart of members' rows as its editor's preview draws them (ChartEditor.vue): in the order asked -- the largest
// value first, the smallest, or by name -- and, when a number is asked, only that many, with no "Other" (the rest
// are not shown). The rows the tile has are all the preview has; Save draws it again from the data.
export function arranged(payload, order, limit) {
  const x = payload.x_key
  const y = (payload.y_keys || [])[0]
  const rest = payload.other ? payload.other.label : null
  const rows = payload.rows || []
  const members = rows.filter((row) => row[x] !== rest)
  const value = (row) => Number(row[y]) || 0
  const by = {
    name: (a, b) => String(a[x]).localeCompare(String(b[x])),
    low: (a, b) => value(a) - value(b),
    high: (a, b) => value(b) - value(a),
  }
  const sorted = [...members].sort(by[order] || by.high)
  const n = Number(limit) || 0
  return n ? { rows: sorted.slice(0, n), other: null }
    : { rows: [...sorted, ...rows.filter((row) => row[x] === rest)], other: payload.other || null }
}
