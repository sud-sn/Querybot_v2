// A panel under a button: open and closed, closed by a press outside it or by Escape (back to its button). It
// opens to the left of its button's right edge when there is no room for it to the right.
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'

function fit(root) {
  const panel = root?.querySelector('.dash-pop')
  if (!panel || getComputedStyle(panel).position === 'fixed') return
  panel.classList.remove('is-end')
  if (panel.getBoundingClientRect().right > document.documentElement.clientWidth - 12) panel.classList.add('is-end')
}

export function usePopover(root, { onOpen } = {}) {
  const open = ref(false)
  const away = (event) => {
    if (root.value && !root.value.contains(event.target)) open.value = false
  }
  const key = (event) => {
    if (event.key !== 'Escape' || !open.value) return
    event.stopPropagation()
    open.value = false
    root.value?.querySelector('[data-trigger]')?.focus()
  }
  const stop = () => {
    document.removeEventListener('pointerdown', away, true)
    document.removeEventListener('keydown', key, true)
  }
  watch(open, (now) => {
    if (!now) return stop()
    document.addEventListener('pointerdown', away, true)
    document.addEventListener('keydown', key, true)
    nextTick(() => fit(root.value))      // placed before anything in it takes the focus
    onOpen?.()
  })
  onBeforeUnmount(stop)
  return open
}
