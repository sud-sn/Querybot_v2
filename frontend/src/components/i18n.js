// The page's own message catalogue and icons (portal_base.html puts both on window): one source for the server's
// markup and these components.
export const t = (id, vars) => (window.qbT ? window.qbT(id, vars) : id)
export const plural = (stem, count, vars) => (window.qbPlural ? window.qbPlural(stem, count, vars) : String(count))
export const icon = (name, size = 16) => (window.qbIcon ? window.qbIcon(name, size) : '')
export const pageLang = () => window.QB_LANG || document.documentElement.lang || 'en'
