// The dashboard page's filter bar (FilterBar.vue), mounted over the plain form the server draws in its place
// (portal_dashboard.html #dashFilterApp), from the state the server put beside it (#dashFilterState).
import { createApp, reactive } from 'vue'
import FilterBar from './FilterBar.vue'

const node = document.getElementById('dashFilterApp')
let state = null
try {
  state = JSON.parse(document.getElementById('dashFilterState')?.textContent || 'null')
} catch {
  state = null
}
if (node && state) {
  const store = reactive({ state, busy: false })
  createApp(FilterBar, { store }).mount(node)
  node.classList.add('is-mounted')
}
