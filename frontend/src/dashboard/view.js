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
  return `/portal/dashboard?${params}`
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
