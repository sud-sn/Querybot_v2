/*
 * static/js/qb-charts.js -- the one chart renderer.
 *
 * Every chart the product draws for an answer is built here: in the chat, in
 * the answer's side pane, full screen, and on a dashboard. The chat and the
 * dashboard used to carry a copy each -- "kept byte-identical", said a comment
 * above one of them -- and the copies drifted: the dashboard never learned the
 * heatmap, the waterfall or the forecast band, drew gradient bars twice as
 * thick with 10px rounded ends, and guessed whether an axis was time from a
 * label regex that read "Marseille" as March.
 *
 * WHICH chart is decided upstream (core/chart_spec.py, carried on the payload
 * as chart_spec); this file only draws it. So an axis is time when the spec
 * says its column is temporal, never because a label happens to contain "mar".
 *
 * Built on Apache ECharts 6 and drawn as SVG, so text and hairlines stay crisp
 * at every zoom and pixel density. The look follows the dataviz method the
 * product validates its palettes with:
 *   - thin, flat marks: bars at most 24px with a 4px rounded data end and a
 *     square baseline, 2px lines, 8px markers ringed in the surface colour, a
 *     10% wash under a single line -- no gradients, glows or shadows;
 *   - touching marks separated by a 2px gap in the surface colour, never by a
 *     drawn border;
 *   - hairline solid gridlines, a recessive axis, text in text tokens and never
 *     in a series colour;
 *   - a legend for two or more series and none for one; values labelled
 *     selectively (bar tips when there are few bars, the end of a single line);
 *   - a crosshair on lines, a per-mark tooltip on bars, the value leading.
 *
 * Needs the shell helpers portal_base.html defines (qbT, qbNum, qbPct, qbMonth,
 * QB_NUM) and chart-palettes.js (QB_PALETTES, QB_CHART_THEME, QB_CHART_STATUS,
 * QB_SEQUENTIAL). Everything else it needs is in this file.
 */
// The leading semicolon keeps this file safe to concatenate after a script
// that ends in an unterminated expression.
;(function (global) {
  'use strict';

  const CATEGORY_CAP = 20;
  const PIE_CAP = 12;
  const MARKER_MAX_POINTS = 24;

  // ── Shell helpers ─────────────────────────────────────────────────────────
  function t(id, vars) {
    return global.qbT ? global.qbT(id, vars) : id;
  }
  function escHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g,
      ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  }
  function num(v) {
    if (v === null || v === undefined || v === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  function money(body, symbol) {
    const sym = String(symbol || '$').trim();
    const fmt = global.QB_NUM || {};
    return fmt.currency_after ? body + (fmt.currency_gap || ' ') + sym : sym + body;
  }

  // The compact ladder for axis ticks and labels. A trillion tier, because
  // 1.2e12 printed "1200B"; the suffixes from the catalogue, because "B" is a
  // thousand times larger in the French long scale ("Md"); significant digits
  // below 0.01, or an axis of rates read as a column of zeros.
  function compactNumber(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v == null ? '' : v);
    const abs = Math.abs(n);
    // min 0: "812K", not "812.0K" -- qbNum's default minimum for a fraction
    // is two places, which max:1 alone cannot trim.
    const tier = (unit, id) => global.qbNum(n / unit, {min: 0, max: 1}) + t('ui.num.compact.' + id);
    if (abs >= 1e12) return tier(1e12, 'trillion');
    if (abs >= 1e9) return tier(1e9, 'billion');
    if (abs >= 1e6) return tier(1e6, 'million');
    if (abs >= 1e3) return tier(1e3, 'thousand');
    if (n === 0) return '0';
    if (abs < 0.01) {
      const kept = Number(n.toPrecision(2));
      const places = (String(kept).split('.')[1] || '').length;
      return global.qbNum(kept, {min: places, max: places, grouping: false});
    }
    return global.qbNum(n, {min: 0, max: 2});
  }

  function normKey(v) {
    return String(v || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
  }
  function rolesOf(payload) {
    return (payload && (payload.column_roles || (payload.chart_spec || {}).column_roles)) || {};
  }
  function roleOf(payload, col) {
    const key = normKey(col);
    const roles = rolesOf(payload);
    for (const raw of Object.keys(roles)) {
      if (normKey(raw) === key) return roles[raw] || {};
    }
    return {};
  }

  function formatFor(payload, col) {
    const key = normKey(col);
    const explicit = (payload && payload.column_formats) || {};
    for (const raw of Object.keys(explicit)) {
      if (normKey(raw) === key) return String(explicit[raw] || 'number').toLowerCase();
    }
    const role = roleOf(payload, col);
    if (role.format) return String(role.format).toLowerCase();
    if (/percent|percentage|pct|rate|ratio|share/i.test(String(col))) return 'percentage';
    if (/revenue|sales|amount|amt|charge|cost|cogs|price|profit|salary|usd|balance/i.test(String(col))) return 'currency';
    return 'number';
  }

  // `fallback` is display copy, returned verbatim; only a real column name goes
  // through the identifier prettifier (total_revenue -> Total Revenue), which
  // is wrong for a translated phrase.
  function columnLabel(payload, col, fallback) {
    const role = roleOf(payload, col);
    if (role.label) return String(role.label);
    if (!col) return fallback == null ? t('ui.chart.value') : String(fallback);
    return String(col).replace(/[_-]+/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
  }

  function formatValue(v, fmt, compact) {
    if (v === null || v === undefined || v === '') return '—';
    const n = Number(v);
    if (Number.isNaN(n)) return String(v);
    // The sign leads the symbol: -$80.2K, never $-80.2K.
    if (fmt === 'currency') {
      const body = compact ? compactNumber(Math.abs(n)) : global.qbNum(Math.abs(n), {min: 2, max: 2});
      return (n < 0 ? '-' : '') + money(body, '$');
    }
    if (fmt === 'percentage') return global.qbNum(n, {max: compact ? 1 : 2}) + (global.QB_NUM || {}).percent_gap + '%';
    return compact ? compactNumber(n) : global.qbNum(n, {max: 4});
  }

  // Which type buttons a reader may press: what the spec says this RESULT can
  // honestly be drawn as. Offering a type the data cannot support ships a
  // button that draws an empty chart.
  function offeredTypes(payload, everyType) {
    const p = payload || {};
    const allowed = Array.isArray(p.renderable_types) && p.renderable_types.length
      ? p.renderable_types
      : (Array.isArray(p.allowed_types) ? p.allowed_types : everyType);
    return everyType.filter(kind => allowed.includes(kind));
  }

  // ── Periods on a time axis ────────────────────────────────────────────────
  // A month key reaches the browser as "202401", a day key as "20240115". Read
  // on an axis those are codes; the reader needs "Jan 2024". Only a column the
  // spec calls temporal is read this way, and a label that is not a period
  // passes through untouched.
  const MONTH_COLUMN_RE = /(^|_)(mth|month|mo|mois)(_|$)/i;
  function periodParts(raw) {
    const s = String(raw == null ? '' : raw).trim();
    let m = /^(\d{4})(\d{2})(\d{2})?$/.exec(s) || /^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?:[T ].*)?$/.exec(s);
    if (!m) return null;
    const year = Number(m[1]), month = Number(m[2]), day = m[3] ? Number(m[3]) : 0;
    if (year < 1900 || year > 2199 || month < 1 || month > 12 || day > 31) return null;
    return {year, month, day};
  }
  function periodLabel(raw, long) {
    const p = periodParts(raw);
    if (!p || !global.qbMonth) return String(raw == null ? '' : raw);
    const month = global.qbMonth(p.month, !long);
    return p.day ? `${p.day} ${month} ${p.year}` : `${month} ${p.year}`;
  }
  // The axis writes the year once, under the first label of each year, so a
  // twelve-month axis reads "Jan / 2024, Feb, Mar ..." instead of repeating
  // the year twelve times.
  function periodAxisFormatter(labels, column) {
    // A month NUMBER on a month column: 1..12 read as Jan..Dec.
    if (MONTH_COLUMN_RE.test(String(column || '')) && labels.length
        && labels.every(v => /^\d{1,2}$/.test(v) && Number(v) >= 1 && Number(v) <= 12) && global.qbMonth) {
      return value => global.qbMonth(Number(value), true);
    }
    const parts = labels.map(periodParts);
    if (!parts.length || parts.some(p => !p)) return null;
    return (value, index) => {
      const p = parts[index] || periodParts(value);
      if (!p) return String(value);
      const prev = index > 0 ? parts[index - 1] : null;
      const month = global.qbMonth(p.month, true);
      const head = p.day ? `${p.day} ${month}` : month;
      return (!prev || prev.year !== p.year) ? `${head}\n${p.year}` : head;
    };
  }

  // ── Theme tokens ──────────────────────────────────────────────────────────
  // ECharts paints concrete strings, never var(); chart-palettes.js resolves the
  // design tokens at call time. A missing key falls back to the token's value.
  function chrome() {
    const theme = (global.QB_CHART_THEME ? global.QB_CHART_THEME() : {}) || {};
    return {
      font: theme.font || "'Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif",
      surface: theme.surface || '#F5F8F6',
      raised: theme.tooltipBg || '#FAFCFB',
      ink: theme.tooltipText || theme.ink || '#161E1A',
      ink2: theme.ink2 || '#45504A',
      muted: theme.axis || '#5A665F',
      grid: theme.split || '#D4DDD8',
      axis: theme.axisLine || '#B9C6C0',
      good: theme.good || '#337438',
      bad: theme.bad || '#A73832',
    };
  }
  function sequentialRamp() {
    return global.QB_SEQUENTIAL || ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  }
  // White or ink on a filled cell, by the fill's luminance, so a label set
  // inside a colour always clears contrast.
  function inkOn(hex, c) {
    const h = String(hex || '').replace('#', '');
    if (h.length !== 6) return c.ink;
    const [r, g, b] = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255)
      .map(v => (v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return lum < 0.36 ? '#FFFFFF' : c.ink;
  }
  function rampColor(ramp, ratio) {
    const r = Math.max(0, Math.min(1, Number(ratio) || 0));
    return ramp[Math.min(ramp.length - 1, Math.round(r * (ramp.length - 1)))];
  }

  // ── Tooltip ───────────────────────────────────────────────────────────────
  // One card style everywhere: the category or period as a quiet header, then
  // one row per series -- a short stroke of the series colour as its key, the
  // name in secondary ink, the value strong and last. The value is what the
  // reader came for. Every label is escaped: series and category names come
  // from the warehouse.
  function tooltipBase(c) {
    return {
      backgroundColor: c.raised,
      borderColor: c.axis,
      borderWidth: 1,
      padding: [8, 10],
      confine: true,
      textStyle: {color: c.ink, fontSize: 12, fontFamily: c.font},
      extraCssText: 'border-radius:5px;box-shadow:0 1px 2px rgba(22,30,26,.06),0 4px 14px rgba(22,30,26,.10);',
    };
  }
  function tipHeader(text, c) {
    return `<div style="color:${c.ink2};font-size:11px;margin-bottom:5px">${escHtml(text)}</div>`;
  }
  function tipRow(color, name, value, c, shape) {
    const key = shape === 'swatch'
      ? `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color}"></span>`
      : `<span style="display:inline-block;width:10px;height:2px;border-radius:1px;background:${color};vertical-align:middle"></span>`;
    return `<div style="display:flex;align-items:center;gap:8px;line-height:18px">${key}`
      + `<span style="color:${c.ink2};flex:1;white-space:nowrap">${escHtml(name)}</span>`
      + `<span style="font-weight:600;color:${c.ink};font-variant-numeric:tabular-nums;margin-left:12px">${value}</span></div>`;
  }

  // ── Annotations ───────────────────────────────────────────────────────────
  // core/insight.py's biggest period drop and gain, anchored on the series' own
  // value at that period so the marker sits on the mark it describes. A small
  // status dot ringed in the surface colour, and a label carrying an arrow and
  // a signed percentage -- never colour alone.
  function annotationMarkPoints(payload, labels, values) {
    const ann = payload && payload.annotations;
    if (!ann) return [];
    const c = chrome();
    const points = [];
    const push = (entry, kind) => {
      if (!entry) return;
      const idx = labels.indexOf(String(entry.period == null ? '' : entry.period));
      if (idx < 0 || values[idx] == null) return;
      const isDrop = kind === 'drop';
      const status = global.QB_CHART_STATUS ? global.QB_CHART_STATUS() : {color: {}, label: {}};
      const pct = Number(entry.pct_change);
      const pctLabel = Number.isFinite(pct) ? `${pct > 0 ? '+' : ''}${global.qbPct(pct, 1)}` : '';
      points.push({
        name: t(isDrop ? 'ui.chat.chart.biggest_drop' : 'ui.chat.chart.biggest_gain'),
        coord: [idx, values[idx]],
        value: pctLabel,
        symbol: 'circle',
        symbolSize: 9,
        itemStyle: {color: status.color[kind] || (isDrop ? c.bad : c.good), borderColor: c.surface, borderWidth: 2},
        label: {
          show: true,
          position: isDrop ? 'bottom' : 'top',
          distance: 6,
          formatter: `${isDrop ? '↓' : '↑'} ${pctLabel} ${t(isDrop ? 'ui.chart.drop' : 'ui.chart.gain')}`,
          fontSize: 11,
          fontWeight: 600,
          color: status.label[kind] || (isDrop ? c.bad : c.good),
          textBorderColor: c.surface,
          textBorderWidth: 3,
        },
      });
    };
    push(ann.biggest_period_drop, 'drop');
    push(ann.biggest_period_gain, 'gain');
    return points;
  }

  // ── The option ────────────────────────────────────────────────────────────
  function buildOption(payload) {
    let rows = Array.isArray(payload && payload.rows) ? payload.rows : [];
    const xKey = payload && payload.x_key;
    let yKeys = Array.isArray(payload && payload.y_keys) ? payload.y_keys.slice() : [];

    const palettes = global.QB_PALETTES || {};
    const colors = palettes[(payload && payload.color_palette) || 'default'] || palettes.default
      || ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
    const c = chrome();

    // A series past the palette's length would repeat a colour -- two series
    // drawn identically, a legend with duplicate swatches. They are dropped
    // and the chart says so.
    let droppedSeries = 0;
    if (yKeys.length > colors.length) {
      droppedSeries = yKeys.length - colors.length;
      yKeys = yKeys.slice(0, colors.length);
    }
    const yKey = yKeys[0];
    const formatOf = col => formatFor(payload, col);
    const valueFmt = (value, col, compact) => formatValue(value, formatOf(col), compact);
    const xLabel = columnLabel(payload, xKey, t('ui.chart.category'));
    const yLabel = columnLabel(payload, yKey, t('ui.chart.value'));
    let labels = rows.map(r => String((xKey && r && r[xKey] != null) ? r[xKey] : ''));

    // Time is what the spec says it is. The regex fallback serves only a chart
    // saved before the spec travelled with the payload.
    const spec = (payload && payload.chart_spec) || {};
    const xRole = (spec.x && spec.x.role) || roleOf(payload, xKey).role;
    const temporal = xRole
      ? xRole === 'temporal'
      : /^(19|20)\d{2}([-/]?\d{2})?|^(q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b/i.test(labels[0] || '');

    const req = String((payload && payload.chart_type) || 'bar').toLowerCase();
    const known = ['pie', 'donut', 'scatter', 'area', 'line', 'waterfall', 'heatmap', 'funnel',
                   'forecast', 'histogram', 'boxplot', 'treemap', 'bar'];
    const type = known.includes(req) ? req : (temporal ? 'line' : 'bar');

    // ── Density ────────────────────────────────────────────────────────────
    // Past a readable count a categorical chart shows the largest values and
    // says so. Never a time series: its order is its meaning, and it gets a
    // zoom instead. A pie keeps its total -- the tail becomes one slice.
    const ordered = type === 'line' || type === 'area' || type === 'forecast' || temporal;
    let truncatedFrom = 0;
    if (!ordered && (type === 'bar' || type === 'pie' || type === 'donut') && yKey) {
      const cap = (type === 'pie' || type === 'donut') ? PIE_CAP : CATEGORY_CAP;
      if (rows.length > cap) {
        truncatedFrom = rows.length;
        // Ranked on the whole bar, not its first segment, so a grouped chart's
        // largest category survives whichever period happens to come first.
        const size = r => yKeys.reduce((s, k) => s + Math.abs(num(r && r[k]) || 0), 0);
        const ranked = rows.slice().sort((a, b) => size(b) - size(a));
        const head = ranked.slice(0, cap);
        if (type === 'pie' || type === 'donut') {
          const rest = ranked.slice(cap).reduce((s, r) => s + (num(r && r[yKey]) || 0), 0);
          head.push({[xKey]: t('ui.chart.other_bucket', {count: global.qbNum(truncatedFrom - cap)}), [yKey]: rest});
        }
        rows = head;
        labels = rows.map(r => String((xKey && r && r[xKey] != null) ? r[xKey] : ''));
      }
    }

    // ── Nothing to draw ────────────────────────────────────────────────────
    if (!rows.length || !yKey) {
      return {
        backgroundColor: 'transparent',
        graphic: {
          type: 'text', left: 'center', top: 'middle',
          style: {
            text: t(rows.length ? 'ui.chat.chart.no_value_column' : 'ui.chat.chart.no_rows'),
            fill: c.muted, fontSize: 13, fontFamily: c.font,
          },
        },
      };
    }

    // A reader not told the chart shows a subset reads it as the whole answer.
    const capParts = [];
    if (truncatedFrom) {
      capParts.push((type === 'pie' || type === 'donut')
        ? t('ui.chart.cap.pie_remainder', {shown: PIE_CAP, total: truncatedFrom})
        : t('ui.chart.cap.largest_of', {shown: CATEGORY_CAP, total: truncatedFrom}));
    }
    if (droppedSeries) capParts.push(t('ui.chart.cap.series_hidden', {count: droppedSeries}));

    const maxLabel = labels.reduce((m, v) => Math.max(m, v.length), 0);
    const longLabels = maxLabel > 14;
    const manyLabels = labels.length > 9;
    const labelFmt = v => { const s = String(v == null ? '' : v); return s.length > 22 ? s.slice(0, 21) + '…' : s; };
    const monthNumbers = temporal && MONTH_COLUMN_RE.test(String(xKey || ''))
      && labels.every(v => /^\d{1,2}$/.test(v) && Number(v) >= 1 && Number(v) <= 12);
    const shownLabel = raw => (monthNumbers && global.qbMonth ? global.qbMonth(Number(raw), false)
      : (temporal ? periodLabel(raw, true) : String(raw)));
    const base = {
      color: colors,
      backgroundColor: 'transparent',
      animationDuration: 300,
      animationEasing: 'cubicOut',
      textStyle: {fontFamily: c.font},
    };
    const legendBase = (names, line) => ({
      type: 'scroll', top: 0, left: 0, right: 0,
      icon: line ? 'path://M0 0h14v2.5H0z' : 'roundRect',
      itemWidth: line ? 14 : 10, itemHeight: line ? 3 : 10, itemGap: 16,
      textStyle: {color: c.ink2, fontSize: 11, fontFamily: c.font},
      pageIconColor: c.ink2, pageIconInactiveColor: c.grid, pageIconSize: 9,
      pageTextStyle: {color: c.muted, fontSize: 10},
      // A series' surface-coloured marker ring would otherwise be inherited
      // by its legend key and swallow a 3px line whole.
      itemStyle: {borderWidth: 0},
      data: names,
    });
    const legendReserve = yKeys.length > 1 ? 26 : 0;
    const capTitle = top => (capParts.length ? {
      text: '', subtext: capParts.join(' · '), left: 0, top,
      padding: 0, itemGap: 0,
      subtextStyle: {color: c.muted, fontSize: 11, fontFamily: c.font},
    } : undefined);
    const capReserve = capParts.length ? 20 : 0;

    // ── Pie / donut ────────────────────────────────────────────────────────
    if (type === 'pie' || type === 'donut') {
      // One slice per category: rows sharing a name are totalled, or one
      // category is drawn twice. Whether the measure may be totalled at all is
      // decided upstream, where the pie was chosen.
      const agg = new Map();
      rows.forEach(r => {
        const name = String((xKey && r && r[xKey] != null) ? r[xKey] : t('ui.chart.unspecified'));
        agg.set(name, (agg.get(name) || 0) + (num(r && r[yKey]) || 0));
      });
      const data = Array.from(agg, ([name, value], i) => ({name, value, itemStyle: {color: colors[i % colors.length]}}));
      const total = data.reduce((s, d) => s + (Number(d.value) || 0), 0);
      const share = v => (total > 0 ? (Number(v) || 0) / total * 100 : 0);
      const shares = new Map(data.map(d => [d.name, share(d.value)]));
      const narrow = (global.innerWidth || 1200) <= 900;
      const donut = type === 'donut';
      // The legend is always there, with each share -- it is the identity
      // channel that does not depend on matching colours. Up to six slices
      // also carry a direct label, and the legend sits below the pie; past
      // six the labels would collide, and the legend beside the pie is the
      // only one.
      const labelled = data.length <= 6;
      const center = labelled ? ['50%', '46%'] : (narrow ? ['50%', '42%'] : ['36%', '52%']);
      return Object.assign(base, {
        title: capTitle(0),
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => tipHeader(`${xLabel}: ${p.name == null ? t('ui.chart.unspecified') : p.name}`, c)
            + tipRow(p.color, yLabel, valueFmt(p.value, yKey), c, 'swatch')
            + tipRow('transparent', t('ui.chart.share_of_total'),
                     global.qbPct(Number(p.percent != null ? p.percent : share(p.value)), 1), c, 'swatch'),
        }),
        legend: Object.assign(legendBase(data.map(d => d.name), false), (labelled || narrow)
          ? {type: 'scroll', orient: 'horizontal', left: 'center', top: 'auto', bottom: 0}
          : {type: 'scroll', orient: 'vertical', left: 'auto', right: 8, top: 'middle'}, {
          formatter: name => `${name.length > 18 ? name.slice(0, 17) + '…' : name}  ${global.qbPct(shares.get(name) || 0, 1)}`,
        }),
        graphic: donut ? [{
          type: 'group', left: center[0], top: center[1],
          bounding: 'raw',
          children: [
            {type: 'text', style: {text: valueFmt(total, yKey, true), fill: c.ink, fontSize: 18, fontWeight: 600,
              fontFamily: c.font, textAlign: 'center', textVerticalAlign: 'bottom'}},
            {type: 'text', top: 4, style: {text: t('ui.chart.total'), fill: c.muted, fontSize: 11,
              fontFamily: c.font, textAlign: 'center', textVerticalAlign: 'top'}},
          ],
        }] : undefined,
        series: [{
          type: 'pie',
          radius: donut ? (labelled ? ['40%', '58%'] : ['50%', '72%']) : ['0%', labelled ? '58%' : '72%'],
          center,
          data,
          // The 2px gap between slices is the surface showing through, which
          // is what separates them -- not a drawn outline.
          itemStyle: {borderColor: c.surface, borderWidth: 2},
          label: {
            show: labelled,
            formatter: p => `{name|${String(p.name || '').slice(0, 24)}}\n{pct|${global.qbPct(Number(p.percent != null ? p.percent : share(p.value)), 1)}}`,
            rich: {
              name: {color: c.ink2, fontSize: 11, fontFamily: c.font, lineHeight: 15},
              pct: {color: c.ink, fontSize: 12, fontWeight: 600, fontFamily: c.font, lineHeight: 16},
            },
          },
          labelLine: {show: labelled, length: 10, length2: 10, lineStyle: {color: c.axis, width: 1}},
          emphasis: {scale: true, scaleSize: 4, label: {fontWeight: 600}},
        }],
      });
    }

    // ── Heatmap ────────────────────────────────────────────────────────────
    // Rows down the side, columns along the top, one sequential hue light to
    // dark. Cells meet at a 2px surface gap; a label sits in a cell only while
    // the grid is small enough to read it, in white or ink by the cell's fill.
    if (type === 'heatmap') {
      const rowKey = xKey || Object.keys(rows[0] || {}).find(k => !k.startsWith('__') && k !== 'cohort_size') || 'cohort';
      const colKeys = (yKeys.length > 1 ? yKeys : Object.keys(rows[0] || {}).filter(
        k => k !== rowKey && k !== 'cohort_size' && !k.startsWith('__')));
      const rowLabels = rows.map(r => String(r[rowKey] == null ? '' : r[rowKey]));
      // A cohort grid holds retention percentages and says so; any other grid
      // is one measure pivoted, and reads in that measure's own format.
      const measure = payload && payload.grouped_measure;
      const cellFmt = (v, compact) => (measure
        ? valueFmt(v, measure, compact)
        : global.qbPct(v, compact ? 0 : 1));
      // [x, y, value] as ECharts passes it: params.value, or the data item
      // itself when that is the bare triple.
      const cellOf = p => {
        const d = p && Array.isArray(p.value) ? p.value
          : (p && Array.isArray(p.data) ? p.data : ((p && p.data && p.data.value) || []));
        return {x: d[0], y: d[1], v: d[2] == null ? null : num(d[2])};
      };
      const values = [];
      rows.forEach(r => colKeys.forEach(k => { const v = num(r[k]); if (v != null) values.push(v); }));
      const lo = values.length ? Math.min(...values) : 0;
      const hi = values.length ? Math.max(...values) : 1;
      const ramp = sequentialRamp();
      const showCells = colKeys.length <= 12 && rows.length <= 16;
      const data = [];
      rows.forEach((r, yi) => colKeys.forEach((k, xi) => {
        const v = num(r[k]);
        const fill = v == null ? c.surface : rampColor(ramp, hi > lo ? (v - lo) / (hi - lo) : 0.5);
        data.push({value: [xi, yi, v], label: {color: inkOn(fill, c)}});
      }));
      return Object.assign(base, {
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => {
            const cell = cellOf(p);
            const col = colKeys[cell.x];
            const abs = rows[cell.y] ? rows[cell.y]['__abs_' + col] : null;
            return tipHeader(`${rowLabels[cell.y] || ''} · ${temporal || !measure ? periodLabel(col, true) : col}`, c)
              + tipRow(p.color, measure ? columnLabel(payload, measure) : t('ui.chart.retention'),
                       cell.v == null ? t('ui.chart.not_available') : cellFmt(cell.v), c, 'swatch')
              + (abs != null ? tipRow('transparent', t('ui.chart.users'), global.qbNum(abs, {max: 0}), c, 'swatch') : '');
          },
        }),
        grid: {left: 8, right: 8, top: 8, bottom: 44, containLabel: true},
        xAxis: {
          type: 'category', data: colKeys, position: 'top',
          axisLabel: {color: c.muted, fontSize: 11, formatter: v => labelFmt(periodParts(v) ? periodLabel(v) : v), hideOverlap: true},
          axisLine: {show: false}, axisTick: {show: false}, splitArea: {show: false},
        },
        yAxis: {
          type: 'category', data: rowLabels, inverse: true,
          axisLabel: {color: c.muted, fontSize: 11, formatter: labelFmt},
          axisLine: {show: false}, axisTick: {show: false}, splitArea: {show: false},
        },
        visualMap: {
          min: lo, max: hi > lo ? hi : lo + 1, calculable: false, orient: 'horizontal',
          left: 'center', bottom: 0, itemWidth: 10, itemHeight: 160,
          inRange: {color: ramp}, text: [cellFmt(hi, true), cellFmt(lo, true)],
          formatter: v => cellFmt(v, true),
          textStyle: {color: c.muted, fontSize: 10, fontFamily: c.font},
        },
        series: [{
          type: 'heatmap', data,
          itemStyle: {borderColor: c.surface, borderWidth: 2},
          label: {show: showCells, fontSize: 10, fontFamily: c.font,
                  formatter: p => { const cell = cellOf(p); return cell.v == null ? '' : cellFmt(cell.v, true); }},
          emphasis: {itemStyle: {borderColor: c.ink, borderWidth: 1}},
        }],
      });
    }

    // ── Waterfall ──────────────────────────────────────────────────────────
    // Each variance floats from the running total. Direction is carried by the
    // product's delta colours AND a sign on every label -- never colour alone.
    if (type === 'waterfall') {
      const varKey = yKeys.find(k => /^variance$|^variance_pct$|^var/i.test(k)) || yKeys[0];
      const deltas = rows.map(r => num(r && r[varKey]) || 0);
      let running = 0;
      const bases = [];
      const spans = [];
      deltas.forEach(v => { bases.push(v >= 0 ? running : running + v); spans.push(Math.abs(v)); running += v; });
      return Object.assign(base, {
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => {
            const v = deltas[p.dataIndex];
            return tipHeader(labels[p.dataIndex], c)
              + tipRow(v >= 0 ? c.good : c.bad, t('ui.chart.waterfall.variance'),
                       (v > 0 ? '+' : '') + valueFmt(v, varKey), c, 'swatch');
          },
        }),
        grid: {left: 8, right: 16, top: 20, bottom: 8, containLabel: true},
        xAxis: {
          type: 'category', data: labels,
          axisLabel: {color: c.muted, fontSize: 11, formatter: labelFmt, hideOverlap: true, rotate: longLabels ? 30 : 0},
          axisLine: {lineStyle: {color: c.axis}}, axisTick: {show: false},
        },
        yAxis: {
          type: 'value',
          axisLabel: {color: c.muted, fontSize: 11, formatter: v => valueFmt(v, varKey, true)},
          splitLine: {lineStyle: {color: c.grid, width: 1}}, axisLine: {show: false}, axisTick: {show: false},
        },
        series: [
          {type: 'bar', stack: 'waterfall', data: bases, silent: true,
           itemStyle: {color: 'transparent'}, tooltip: {show: false}, barMaxWidth: 24},
          {type: 'bar', stack: 'waterfall', barMaxWidth: 24,
           data: spans.map((v, i) => ({
             value: v,
             itemStyle: {color: deltas[i] >= 0 ? c.good : c.bad, borderRadius: deltas[i] >= 0 ? [4, 4, 0, 0] : [0, 0, 4, 4]},
             label: {position: deltas[i] >= 0 ? 'top' : 'bottom'},
           })),
           label: {
             show: rows.length <= 16, color: c.ink2, fontSize: 11, fontFamily: c.font,
             formatter: p => { const v = deltas[p.dataIndex]; return (v > 0 ? '+' : '') + valueFmt(v, varKey, true); },
           },
          },
        ],
      });
    }

    // ── Funnel ─────────────────────────────────────────────────────────────
    // Ordered stages take an ordinal ramp of one hue, darkest first, with a
    // 2px surface gap between stages.
    if (type === 'funnel') {
      const meta = ['funnel_pct', 'conversion_rate', 'drop_off', 'cumulative_conversion'];
      const stageKey = xKey || Object.keys(rows[0] || {}).find(k => !meta.includes(k)) || 'stage';
      const countKey = yKey || yKeys.find(k => !meta.includes(k)) || yKeys[0];
      const ramp = sequentialRamp().slice(2).reverse();
      const data = rows.map((r, i) => {
        const fill = ramp[Math.min(i, ramp.length - 1)];
        return {
          name: String(r[stageKey] == null ? '' : r[stageKey]),
          value: num(r.funnel_pct != null ? r.funnel_pct : r[countKey]) || 0,
          itemStyle: {color: fill},
          label: {color: inkOn(fill, c)},
          __count: num(r[countKey]), __conv: r.conversion_rate, __drop: r.drop_off,
        };
      });
      return Object.assign(base, {
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => {
            const d = data[p.dataIndex] || {};
            return tipHeader(p.name, c)
              + tipRow(p.color, t('ui.chart.count'), valueFmt(d.__count, countKey), c, 'swatch')
              + (d.__conv != null ? tipRow('transparent', t('ui.chart.from_prev', {pct: ''}).trim(), global.qbPct(d.__conv, 1), c, 'swatch') : '')
              + (d.__drop != null && d.__drop > 0 ? tipRow('transparent', t('ui.chart.dropoff'), valueFmt(d.__drop, countKey), c, 'swatch') : '');
          },
        }),
        series: [{
          type: 'funnel', left: '8%', width: '84%', top: 12, bottom: 12,
          min: 0, max: 100, minSize: '6%', maxSize: '100%', sort: 'descending', gap: 2,
          data,
          label: {show: true, position: 'inside', fontSize: 12, fontFamily: c.font,
                  formatter: p => `${p.name}\n${global.qbPct(p.value, 1)}`},
          itemStyle: {borderWidth: 0},
        }],
      });
    }

    // ── Forecast ───────────────────────────────────────────────────────────
    if (type === 'forecast') {
      const meta = ['is_forecast', 'forecast_value', 'forecast_low', 'forecast_high',
                    '__trend_slope', '__trend_r2', '__forecast_meta'];
      const periodKey = xKey || Object.keys(rows[0] || {}).find(k => !meta.includes(k) && typeof rows[0][k] === 'string') || 'period';
      const metricKey = yKey || yKeys.find(k => !meta.includes(k) && typeof rows[0][k] === 'number') || yKeys[0];
      const periods = rows.map(r => String(r[periodKey] == null ? '' : r[periodKey]));
      const hist = rows.map(r => (r.is_forecast ? null : num(r[metricKey])));
      const proj = rows.map(r => (r.is_forecast ? num(r[metricKey]) : null));
      // The projection starts at the last actual point, so it reads as a
      // continuation rather than a second line floating to the right.
      const lastActual = rows.reduce((acc, r, i) => (r.is_forecast ? acc : i), -1);
      if (lastActual >= 0) proj[lastActual] = num(rows[lastActual][metricKey]);

      const fit = (payload && payload.forecast_meta) || {};
      const slope = fit.slope != null ? fit.slope : (rows[0] || {}).__trend_slope;
      const r2 = fit.r2 != null ? fit.r2 : (rows[0] || {}).__trend_r2;
      // A slope is captioned only when the line explains something.
      const trendNote = (slope != null && r2 != null)
        ? (r2 >= 0.5
          ? t('ui.chart.forecast.trend', {slope: (slope >= 0 ? '+' : '') + valueFmt(slope, metricKey, true),
                                          r2: global.qbNum(r2, {min: 2, max: 2})})
          : t('ui.chart.forecast.no_trend', {r2: global.qbNum(r2, {min: 2, max: 2})}))
        : '';

      // The 95% interval: a hidden baseline plus a shaded span, stacked with
      // stackStrategy 'all' -- the default stacks a value onto the running total
      // only when the signs agree, so a band whose lower bound goes negative
      // would detach and draw from the axis. It opens from zero width at the
      // last actual point, because a period already past is not uncertain.
      const hasBand = rows.some(r => r.is_forecast && r.forecast_low != null);
      const anchor = lastActual >= 0 ? num(rows[lastActual][metricKey]) : null;
      const bandBase = rows.map((r, i) => (anchor != null && i === lastActual ? anchor : (r.is_forecast ? num(r.forecast_low) : null)));
      const bandSpan = rows.map((r, i) => {
        if (anchor != null && i === lastActual) return 0;
        if (!r.is_forecast) return null;
        const lo = num(r.forecast_low), hi = num(r.forecast_high);
        return (lo == null || hi == null) ? null : hi - lo;
      });
      const actualName = t('ui.chat.chart.actual');
      const forecastName = t('ui.chat.chart.forecast');
      const bandName = t('ui.chat.chart.interval');
      const projColor = colors[1] || colors[0];
      const axisFmt = periodAxisFormatter(periods, periodKey);
      return Object.assign(base, {
        color: [colors[0], projColor],
        title: trendNote ? {text: '', subtext: trendNote, left: 0, top: 22, padding: 0,
                            subtextStyle: {color: c.muted, fontSize: 11, fontFamily: c.font}} : undefined,
        // Each key mirrors its mark: a solid line, a dashed one, a swatch for
        // the shaded interval.
        legend: legendBase([
          {name: actualName, icon: 'path://M0 0h14v2.5H0z'},
          {name: forecastName, icon: 'path://M0 0h5v2.5H0zM9 0h5v2.5H9z'},
        ].concat(hasBand ? [{name: bandName, icon: 'roundRect', itemStyle: {opacity: 0.35}}] : []), true),
        grid: {left: 8, right: 16, top: trendNote ? 60 : 36, bottom: 8, containLabel: true},
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'axis',
          axisPointer: {type: 'line', lineStyle: {color: c.axis, width: 1}},
          formatter: params => {
            const arr = Array.isArray(params) ? params : [params];
            const i = arr[0] ? arr[0].dataIndex : 0;
            const row = rows[i] || {};
            const lines = arr.filter(p => p.value != null && p.seriesName !== 'band-base' && p.seriesName !== bandName)
              .map(p => tipRow(p.color, p.seriesName, valueFmt(p.value, metricKey), c));
            if (row.is_forecast && row.forecast_low != null && row.forecast_high != null) {
              lines.push(tipRow(projColor, bandName,
                `${valueFmt(num(row.forecast_low), metricKey)} – ${valueFmt(num(row.forecast_high), metricKey)}`, c, 'swatch'));
            }
            return tipHeader(periodLabel(periods[i], true) + (row.is_forecast ? ` · ${forecastName}` : ''), c) + lines.join('');
          },
        }),
        xAxis: {
          type: 'category', data: periods, boundaryGap: false,
          axisLabel: {color: c.muted, fontSize: 11, hideOverlap: true, formatter: axisFmt || labelFmt},
          axisLine: {lineStyle: {color: c.axis}}, axisTick: {show: false},
        },
        yAxis: {
          // A forecast's subject is a band a few percent wide; anchored at zero
          // it would draw as a flat line. The caption carries the honesty.
          type: 'value', scale: true,
          axisLabel: {color: c.muted, fontSize: 11, formatter: v => valueFmt(v, metricKey, true)},
          splitLine: {lineStyle: {color: c.grid, width: 1}}, axisLine: {show: false}, axisTick: {show: false},
        },
        series: [
          {name: actualName, type: 'line', data: hist, showSymbol: hist.filter(v => v != null).length <= MARKER_MAX_POINTS,
           symbol: 'circle', symbolSize: 8, lineStyle: {width: 2, color: colors[0], cap: 'round', join: 'round'},
           itemStyle: {color: colors[0], borderColor: c.surface, borderWidth: 2},
           areaStyle: {color: colors[0], opacity: 0.10}, connectNulls: false},
          {name: forecastName, type: 'line', data: proj, symbol: 'circle', symbolSize: 8,
           lineStyle: {width: 2, color: projColor, type: [6, 4], cap: 'round'},
           itemStyle: {color: projColor, borderColor: c.surface, borderWidth: 2}, connectNulls: true},
        ].concat(hasBand ? [
          {name: 'band-base', type: 'line', stack: 'fcband', stackStrategy: 'all', data: bandBase,
           lineStyle: {opacity: 0}, itemStyle: {opacity: 0}, symbol: 'none', silent: true,
           tooltip: {show: false}, legendHoverLink: false},
          {name: bandName, type: 'line', stack: 'fcband', stackStrategy: 'all', data: bandSpan,
           lineStyle: {opacity: 0}, symbol: 'none', silent: true, itemStyle: {color: projColor},
           areaStyle: {color: projColor, opacity: 0.14}},
        ] : []),
      });
    }

    // ── Histogram ──────────────────────────────────────────────────────────
    // Bins touch, separated by the 2px surface gap.
    if (type === 'histogram') {
      const bins = rows.map(r => String(r.bin_label == null ? '' : r.bin_label));
      const counts = rows.map(r => num(r.count) || 0);
      const total = counts.reduce((s, v) => s + v, 0);
      return Object.assign(base, {
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => tipHeader(bins[p.dataIndex], c)
            + tipRow(p.color, t('ui.chart.count'), global.qbNum(p.value), c, 'swatch')
            + tipRow('transparent', t('ui.chart.share_of_total'),
                     global.qbPct(total > 0 ? p.value / total * 100 : 0, 1), c, 'swatch'),
        }),
        grid: {left: 8, right: 16, top: 16, bottom: 8, containLabel: true},
        xAxis: {type: 'category', data: bins,
                axisLabel: {color: c.muted, fontSize: 11, hideOverlap: true, formatter: labelFmt},
                axisLine: {lineStyle: {color: c.axis}}, axisTick: {show: false}},
        yAxis: {type: 'value', name: t('ui.chart.count'), nameTextStyle: {color: c.muted, fontSize: 11, align: 'left'},
                axisLabel: {color: c.muted, fontSize: 11, formatter: v => compactNumber(v)},
                splitLine: {lineStyle: {color: c.grid, width: 1}}, axisLine: {show: false}, axisTick: {show: false}},
        series: [{type: 'bar', data: counts, barCategoryGap: 2,
                  itemStyle: {color: colors[0], borderRadius: [2, 2, 0, 0]}}],
      });
    }

    // ── Box plot ───────────────────────────────────────────────────────────
    if (type === 'boxplot') {
      const groups = rows.map(r => String(r.group == null ? '' : r.group));
      const boxes = rows.map(r => (Array.isArray(r.bp_data) ? r.bp_data : [r.bp_min, r.bp_q1, r.bp_median, r.bp_q3, r.bp_max]));
      const outliers = [];
      rows.forEach((r, gi) => (r.bp_outliers || []).forEach(v => outliers.push([gi, v])));
      return Object.assign(base, {
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => {
            if (p.seriesType === 'scatter') return tipHeader(t('ui.chat.chart.outlier'), c) + tipRow(p.color, yLabel, valueFmt(p.value[1], yKey), c, 'swatch');
            const r = rows[p.dataIndex] || {};
            return tipHeader(groups[p.dataIndex], c)
              + tipRow('transparent', t('ui.chart.box.max'), valueFmt(r.bp_max, yKey), c, 'swatch')
              + tipRow('transparent', 'Q3', valueFmt(r.bp_q3, yKey), c, 'swatch')
              + tipRow(p.color, t('ui.chart.box.median'), valueFmt(r.bp_median, yKey), c, 'swatch')
              + tipRow('transparent', 'Q1', valueFmt(r.bp_q1, yKey), c, 'swatch')
              + tipRow('transparent', t('ui.chart.box.min'), valueFmt(r.bp_min, yKey), c, 'swatch')
              + tipRow('transparent', t('ui.chart.box.mean'), valueFmt(r.bp_mean, yKey), c, 'swatch')
              + tipRow('transparent', 'n', global.qbNum(r.bp_count), c, 'swatch');
          },
        }),
        grid: {left: 8, right: 16, top: 16, bottom: 8, containLabel: true},
        xAxis: {type: 'category', data: groups,
                axisLabel: {color: c.muted, fontSize: 11, formatter: labelFmt, hideOverlap: true},
                axisLine: {lineStyle: {color: c.axis}}, axisTick: {show: false}},
        yAxis: {type: 'value', axisLabel: {color: c.muted, fontSize: 11, formatter: v => valueFmt(v, yKey, true)},
                splitLine: {lineStyle: {color: c.grid, width: 1}}, axisLine: {show: false}, axisTick: {show: false}},
        series: [
          {type: 'boxplot', data: boxes, boxWidth: [8, 24],
           itemStyle: {color: colors[0] + '1A', borderColor: colors[0], borderWidth: 1.5}},
        ].concat(outliers.length ? [{type: 'scatter', data: outliers, symbolSize: 7,
          itemStyle: {color: colors[0], borderColor: c.surface, borderWidth: 1.5}}] : []),
      });
    }

    // ── Treemap ────────────────────────────────────────────────────────────
    // Area carries the value, so every tile wears one colour; a hue per tile
    // would cycle the palette past eight.
    if (type === 'treemap') {
      const nameKey = xKey || Object.keys(rows[0] || {}).find(k => typeof rows[0][k] === 'string') || '';
      const sizeKey = yKey || '';
      const tiles = rows.map(r => ({name: String(r[nameKey] == null ? '' : r[nameKey]), value: Math.abs(num(r[sizeKey]) || 0)}))
        .filter(d => d.value > 0);
      const total = tiles.reduce((s, d) => s + d.value, 0);
      return Object.assign(base, {
        tooltip: Object.assign(tooltipBase(c), {
          formatter: p => tipHeader(p.name, c)
            + tipRow(colors[0], columnLabel(payload, sizeKey), valueFmt(p.value, sizeKey), c, 'swatch')
            + tipRow('transparent', t('ui.chart.share_of_total'), global.qbPct(total > 0 ? p.value / total * 100 : 0, 1), c, 'swatch'),
        }),
        series: [{
          type: 'treemap', data: tiles, width: '100%', height: '100%', roam: false, nodeClick: false,
          breadcrumb: {show: false},
          itemStyle: {color: colors[0], borderColor: c.surface, borderWidth: 2, gapWidth: 2},
          label: {show: true, color: inkOn(colors[0], c), fontSize: 11, fontFamily: c.font, overflow: 'truncate',
                  formatter: p => `${p.name}\n${global.qbPct(total > 0 ? p.value / total * 100 : 0, 1)}`},
          emphasis: {itemStyle: {borderColor: c.ink, borderWidth: 1}},
        }],
      });
    }

    // ── Cartesian: scatter, line, area, bar ───────────────────────────────
    const horizontal = type === 'bar' && !temporal && (longLabels || manyLabels);
    const hasAnnotations = Boolean(payload && payload.annotations
      && (payload.annotations.biggest_period_drop || payload.annotations.biggest_period_gain));
    const zoom = ordered && labels.length > 60;
    const axisFmt = temporal ? periodAxisFormatter(labels, xKey) : null;
    const top = legendReserve + capReserve + (hasAnnotations ? 22 : 8);

    if (type === 'scatter') {
      const y2 = yKeys[1];
      const pointLabel = r => String(xKey ? (r && r[xKey] != null ? r[xKey] : '') : '');
      const valueAxis = (col, withName) => ({
        type: 'value', name: withName ? columnLabel(payload, col) : undefined,
        nameLocation: 'middle', nameGap: 28,
        nameTextStyle: {color: c.ink2, fontSize: 11, fontFamily: c.font},
        axisLabel: {color: c.muted, fontSize: 11, formatter: v => valueFmt(v, col, true)},
        splitLine: {lineStyle: {color: c.grid, width: 1}}, axisLine: {show: false}, axisTick: {show: false},
      });
      return Object.assign(base, {
        title: capTitle(0),
        grid: {left: 16, right: 16, top: 12 + capReserve, bottom: 28, containLabel: true},
        tooltip: Object.assign(tooltipBase(c), {
          trigger: 'item',
          formatter: p => tipHeader(`${xLabel}: ${p.value && p.value[2] !== '' ? p.value[2] : t('ui.chart.unspecified')}`, c)
            + tipRow(p.color, columnLabel(payload, yKey), valueFmt(p.value && p.value[0], yKey), c, 'swatch')
            + tipRow(p.color, columnLabel(payload, y2), valueFmt(p.value && p.value[1], y2), c, 'swatch'),
        }),
        xAxis: valueAxis(yKey, true),
        yAxis: Object.assign(valueAxis(y2, true), {nameLocation: 'end', nameGap: 10,
                                                   nameTextStyle: {color: c.ink2, fontSize: 11, align: 'left'}}),
        series: [{
          type: 'scatter', symbolSize: 9,
          data: rows.map(r => [num(r && r[yKey]), num(r && r[y2]), pointLabel(r)]),
          itemStyle: {color: colors[0], opacity: 0.9, borderColor: c.surface, borderWidth: 2},
          emphasis: {scale: 1.4},
        }],
      });
    }

    const categoryAxis = {
      type: 'category',
      data: labels,
      boundaryGap: type === 'bar',
      axisLabel: {
        color: c.muted, fontSize: 11, hideOverlap: true, lineHeight: 14,
        interval: (type === 'bar' && labels.length <= 16) || (!temporal && !manyLabels) ? 0 : 'auto',
        rotate: !horizontal && !temporal && longLabels ? 30 : 0,
        formatter: axisFmt || labelFmt,
      },
      axisLine: {lineStyle: {color: c.axis, width: 1}},
      axisTick: {show: false},
    };
    const valueAxis = {
      type: 'value',
      axisLabel: {color: c.muted, fontSize: 11, formatter: v => valueFmt(v, yKey, true)},
      splitLine: {lineStyle: {color: c.grid, width: 1, type: 'solid'}},
      axisLine: {show: false},
      axisTick: {show: false},
    };
    const option = Object.assign(base, {
      title: capTitle(legendReserve ? 24 : 0),
      grid: horizontal
        ? {left: 8, right: 48, top, bottom: 8, containLabel: true}
        : {left: 8, right: (type === 'line' || type === 'area') && yKeys.length === 1 ? 56 : 16,
           top, bottom: zoom ? 34 : 8, containLabel: true},
      xAxis: horizontal ? valueAxis : categoryAxis,
      // A ranking reads top-down: the first row -- the largest, in the order
      // the query ranked it -- sits at the top of a horizontal bar chart.
      yAxis: horizontal
        ? Object.assign({}, categoryAxis, {inverse: true, axisLabel: Object.assign({}, categoryAxis.axisLabel, {
            rotate: 0, interval: 0, hideOverlap: false, width: 180, overflow: 'truncate', ellipsis: '…',
            formatter: v => String(v == null ? '' : v)})})
        : valueAxis,
      dataZoom: zoom ? [
        {type: 'inside', throttle: 50},
        {type: 'slider', height: 14, bottom: 4, borderColor: 'transparent', backgroundColor: c.grid,
         fillerColor: 'rgba(42,120,214,0.12)', dataBackground: {lineStyle: {color: c.axis}, areaStyle: {color: c.grid}},
         handleStyle: {color: c.raised, borderColor: c.axis}, moveHandleSize: 4,
         textStyle: {color: c.muted, fontSize: 10}},
      ] : undefined,
      legend: yKeys.length > 1
        ? Object.assign(legendBase(yKeys, type === 'line'), {formatter: name => columnLabel(payload, name)})
        : undefined,
      series: [],
    });

    const multiTip = params => {
      const arr = Array.isArray(params) ? params : [params];
      const head = arr[0] ? (arr[0].axisValue != null ? arr[0].axisValue : arr[0].name) : '';
      return tipHeader(shownLabel(head), c)
        + arr.map(p => tipRow(p.color, columnLabel(payload, p.seriesName || ''), valueFmt(p.value, p.seriesName), c,
                              type === 'bar' ? 'swatch' : 'line')).join('');
    };

    // ── Line / area ────────────────────────────────────────────────────────
    if (type === 'line' || type === 'area') {
      const single = yKeys.length === 1;
      option.tooltip = Object.assign(tooltipBase(c), {
        trigger: 'axis',
        // The crosshair finds the period; the reader never has to land on a
        // 2px line to read it.
        axisPointer: {type: 'line', lineStyle: {color: c.axis, width: 1}},
        formatter: multiTip,
      });
      option.series = yKeys.map((k, i) => {
        const values = rows.map(r => num(r && r[k]));
        const color = colors[i % colors.length];
        return {
          name: k, type: 'line', data: values,
          // Straight segments: a smoothed curve bulges past the data and draws
          // values that were never measured.
          smooth: false,
          showSymbol: rows.length <= MARKER_MAX_POINTS,
          symbol: 'circle', symbolSize: 8,
          lineStyle: {width: 2, color, cap: 'round', join: 'round'},
          itemStyle: {color, borderColor: c.surface, borderWidth: 2},
          emphasis: {focus: single ? 'none' : 'series', lineStyle: {width: 2}},
          // An area is a wash under the line, not a block; under several
          // lines it is fainter still, or the washes tint each other into mud.
          areaStyle: type === 'area' ? {color, opacity: single ? 0.10 : 0.06} : undefined,
          // A single line says its latest value at its end.
          endLabel: single ? {show: true, color: c.ink2, fontSize: 11, fontFamily: c.font, distance: 6,
                              formatter: p => valueFmt(p.value, k, true)} : undefined,
          markPoint: i === 0 ? {silent: true, data: annotationMarkPoints(payload, labels, values)} : undefined,
        };
      });
      return option;
    }

    // ── Bar (default) ──────────────────────────────────────────────────────
    const multi = yKeys.length > 1;
    option.tooltip = Object.assign(tooltipBase(c), multi
      ? {trigger: 'axis', axisPointer: {type: 'shadow', shadowStyle: {color: 'rgba(22,30,26,0.05)'}}, formatter: multiTip}
      : {trigger: 'item', formatter: p => tipHeader(`${xLabel}: ${shownLabel(p.name)}`, c)
          + tipRow(p.color, yLabel, valueFmt(p.value, yKey), c, 'swatch')});
    // A variance or a change carries its direction in the product's delta
    // colours, and in a sign on its label; any other single measure keeps one
    // colour for every bar -- a hue per bar would encode the length twice.
    const firstValues = rows.map(r => num(r && r[yKey]));
    const mixedSigns = firstValues.some(v => v != null && v < 0) && firstValues.some(v => v != null && v > 0);
    const delta = !multi && mixedSigns && /(variance|^var_|_var$|change|delta|diff|gap|growth|chg)/i.test(String(yKey));
    const labelled = !multi && rows.length <= (horizontal ? 20 : 12);
    // A labelled bar needs room past its tip: with the axis ending exactly at
    // the largest value, that bar's label was clipped by the chart's edge.
    if (labelled) {
      (horizontal ? option.xAxis : option.yAxis).boundaryGap = [0, '8%'];
    }
    option.series = yKeys.map((k, i) => {
      const color = colors[i % colors.length];
      const values = rows.map(r => num(r && r[k]));
      // The 4px rounded end is the DATA end: for a bar below zero that is its
      // bottom (or its left), not the baseline it grows from.
      const end = v => (horizontal
        ? (v != null && v < 0 ? [4, 0, 0, 4] : [0, 4, 4, 0])
        : (v != null && v < 0 ? [0, 0, 4, 4] : [4, 4, 0, 0]));
      const place = v => (horizontal ? (v != null && v < 0 ? 'left' : 'right') : (v != null && v < 0 ? 'bottom' : 'top'));
      return {
        name: k, type: 'bar',
        barMaxWidth: 24,
        barGap: '12%',
        itemStyle: {color, borderRadius: end(1)},
        // A plain value unless the bar needs its own styling: a variance's
        // direction colour, or a rounded end and a label on the far side of
        // the baseline for a value below zero.
        data: values.map(v => ((delta || (v != null && v < 0)) ? {
          value: v,
          itemStyle: {color: delta ? (v != null && v < 0 ? c.bad : c.good) : color, borderRadius: end(v)},
          label: {position: place(v)},
        } : v)),
        // Bars -> the value at the tip; columns -> on the cap. A bar below
        // zero carries its own position on the far side of the baseline.
        label: labelled ? {
          show: true, position: horizontal ? 'right' : 'top',
          color: c.ink2, fontSize: 11, fontFamily: c.font, distance: 4,
          formatter: p => (delta && p.value > 0 ? '+' : '') + valueFmt(p.value, k, true),
        } : undefined,
        emphasis: {focus: multi ? 'series' : 'none', itemStyle: {opacity: 0.9}},
        markPoint: (i === 0 && !horizontal) ? {silent: true, data: annotationMarkPoints(payload, labels, values)} : undefined,
      };
    });
    return option;
  }

  // ── Mounting ──────────────────────────────────────────────────────────────
  // opts.grow: false keeps the element's height -- a dashboard card has the
  // size its owner gave it on the grid.
  function render(el, payload, opts) {
    if (!el || !payload || !global.echarts) return null;
    const echarts = global.echarts;
    const existing = echarts.getInstanceByDom(el);
    if (existing) existing.dispose();
    const option = buildOption(payload);
    // A horizontal ranking needs a readable row per category. In a fixed-height
    // card twenty categories got 12px each against an 11px label, and the
    // axis hid every other name. The chart grows to fit instead -- a chart
    // whose labels are missing is not a chart of those categories. Where it
    // may not grow, the axis names every other category rather than
    // overprinting them.
    const yAxis = option.yAxis;
    if (yAxis && !Array.isArray(yAxis) && yAxis.type === 'category' && yAxis.inverse && Array.isArray(yAxis.data)) {
      const need = yAxis.data.length * 22 + 72;
      if (opts && opts.grow === false) {
        if (el.clientHeight && need > el.clientHeight) yAxis.axisLabel.interval = 'auto';
      } else if (el.clientHeight && need > el.clientHeight) {
        el.style.height = need + 'px';
      }
    }
    const chart = echarts.init(el, null, {renderer: 'svg'});
    chart.setOption(option, true);
    if (global.ResizeObserver) {
      if (el._qbRO) el._qbRO.disconnect();
      const ro = new global.ResizeObserver(() => { try { chart.resize(); } catch (e) { /* disposed */ } });
      ro.observe(el);
      el._qbRO = ro;
    }
    return chart;
  }

  global.QBCharts = {
    buildOption,
    render,
    offeredTypes,
    formatFor,
    formatValue,
    columnLabel,
    compactNumber,
    periodLabel,
    annotationMarkPoints,
  };
})(typeof window !== 'undefined' ? window : this);
