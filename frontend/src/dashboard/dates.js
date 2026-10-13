// Days as ISO strings ("2026-10-13"), worked out in UTC so no time zone moves a day. Weeks start on Monday, as
// the new core counts them (core2/resolve/time.py).
const pad = (n) => String(n).padStart(2, '0')

export function parse(day) {
  const [y, m, d] = String(day || '').split('-').map(Number)
  return y && m && d ? new Date(Date.UTC(y, m - 1, d)) : null
}
export const iso = (date) => `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`
export const monthStart = (day) => `${String(day).slice(0, 7)}-01`

export function addDays(day, n) {
  const date = parse(day)
  date.setUTCDate(date.getUTCDate() + n)
  return iso(date)
}

// The same day ``n`` months on, kept inside the month it lands in (31 Jan + 1 month is 28 or 29 Feb).
export function addMonths(day, n) {
  const date = parse(day)
  const target = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth() + n, 1))
  const last = new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0)).getUTCDate()
  target.setUTCDate(Math.min(date.getUTCDate(), last))
  return iso(target)
}

// A month as rows of seven days, Monday first; null outside the month.
export function weeks(month) {
  const first = parse(monthStart(month))
  const year = first.getUTCFullYear()
  const index = first.getUTCMonth()
  const count = new Date(Date.UTC(year, index + 1, 0)).getUTCDate()
  const cells = Array((first.getUTCDay() + 6) % 7).fill(null)
  for (let d = 1; d <= count; d += 1) cells.push(iso(new Date(Date.UTC(year, index, d))))
  while (cells.length % 7) cells.push(null)
  const rows = []
  for (let i = 0; i < cells.length; i += 7) rows.push(cells.slice(i, i + 7))
  return rows
}

const format = (date, lang, options) =>
  new Intl.DateTimeFormat(lang, { ...options, timeZone: 'UTC' }).format(date)

export const monthTitle = (month, lang) => format(parse(monthStart(month)), lang, { month: 'long', year: 'numeric' })
// Monday to Sunday: 1 January 2024 was a Monday.
export const weekdays = (lang) =>
  [0, 1, 2, 3, 4, 5, 6].map((i) => ({
    short: format(new Date(Date.UTC(2024, 0, 1 + i)), lang, { weekday: 'narrow' }),
    long: format(new Date(Date.UTC(2024, 0, 1 + i)), lang, { weekday: 'long' }),
  }))
export const dayLabel = (day, lang) =>
  format(parse(day), lang, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })

// "1–13 Oct 2026", "1 Jan – 31 Mar 2026", "1 Dec 2025 – 31 Jan 2026", as the server writes a span
// (portal/dashboard_view.py span_label).
export function spanLabel(start, end, lang) {
  const a = parse(start)
  const b = parse(end)
  if (!a || !b) return ''
  const month = (date) => format(date, lang, { month: 'short' })
  const day = (date, year = true) => `${date.getUTCDate()} ${month(date)}${year ? ` ${date.getUTCFullYear()}` : ''}`
  if (start === end) return day(a)
  if (start.slice(0, 7) === end.slice(0, 7)) return `${a.getUTCDate()}–${day(b)}`
  if (start.slice(0, 4) === end.slice(0, 4)) return `${day(a, false)} – ${day(b)}`
  return `${day(a)} – ${day(b)}`
}
