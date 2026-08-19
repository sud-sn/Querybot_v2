/*
 * Chart series palettes, shared by the portal chat and the portal dashboard.
 *
 * These are deliberately literal hex values, not design tokens: ECharts paints
 * to a canvas and cannot resolve var(), and a categorical series set is a
 * designed qualitative ramp rather than a semantic colour.
 *
 * One array per palette: the product is light only, so the second mode each of
 * these used to carry has been removed along with the theme it served.
 *
 * This file exists because both pages used to hold their own copy of the table,
 * synchronised by a comment. Tuning one and not the other would have drawn the
 * same series in different colours on adjacent pages.
 *
 * The validation rationale for the specific values follows, moved verbatim from
 * portal_chat.html.
 */
// ── Named colour palettes ─────────────────────────────────────────────────────
// Mode-aware: each theme carries separate light/dark hex sets, validated with
// the dataviz skill's scripts/validate_palette.js (OKLCH lightness band,
// chroma floor, CVD-simulated adjacent separation, unsimulated normal-vision
// floor, contrast vs the chart surface) against this app's actual chart
// surfaces (#ffffff light / #0f172a dark), not the validator's generic
// defaults. The previous single-array-for-both-modes palettes all failed
// the validator outright (near-duplicate adjacent colors, several outside
// the lightness/chroma bands) - every set below was re-picked and/or
// re-ordered to clear the checks; ordering matters as much as the hex
// values since adjacency is what the CVD/normal-vision checks measure.
//
// `sunset` and `forest` are the two exceptions: both are inherently narrow
// hue-family themes (warm-only, green-only) sitting exactly on the
// red/green/orange axis that protan/deutan colorblindness confuses most -
// no reordering or re-hex clears the 15.0 unsimulated-ΔE floor for all 8
// adjacent pairs simultaneously in both modes (best achieved: sunset ~14.5,
// forest ~9.4-9.8, vs the 15.0 hard gate). Ship them as documented "fewer
// safe series" options - fine for 2-4 series with the app's existing table
// view / direct labels as the required relief, risky past that.
window.QB_PALETTES = {
  default: ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'],
  ocean: ['#14B8A6','#1D4ED8','#0EA5E9','#2347B8','#0D9488','#2563EB','#38BDF8','#0369A1'],
  // Best-achievable, not fully passing -- see comment above.
  sunset: ['#F97316','#9A3412','#F59E0B','#DC2626','#EC4899','#C2410C','#A21CAF','#E11D48'],
  // Best-achievable, not fully passing -- see comment above.
  forest: ['#10B981','#00704E','#22C55E','#4D7C0F','#059669','#84CC16','#166534','#2FBF8E'],
  candy: ['#EC4899','#CA8A04','#8B5CF6','#14B8A6','#6366F1','#F97316','#0891B2','#DC2626'],
  // Reclassified as a one-hue ORDINAL ramp, not an 8-slot categorical set --
  // true grayscale has ~0 OKLCH chroma, which fails the categorical chroma
  // floor by design (a hue-based check can't apply to a hue-less palette).
  // Validated instead with the skill's ordinal checks: monotone lightness,
  // >=0.06 ΔL between adjacent steps, light/dark-end contrast against the
  // surface, single-hue spread. Use for genuinely ORDERED series (ranked
  // tiers, funnel stages) where a light-to-dark reading carries the order,
  // not for arbitrary unordered category identity.
  mono: ['#a3acb8','#8a95a3','#69748a','#556173','#3d495c','#293344','#151d2b'],
};

// Gradient pair: [bright top, muted bottom] for each palette primary
window.QB_PALETTE_GRADIENTS = {
  default:  ['#4A96E8','#1A68C6'],
  ocean:    ['#38BDF8','#0369A1'],
  sunset:   ['#FB923C','#DC2626'],
  forest:   ['#34D399','#047857'],
  candy:    ['#A78BFA','#6D28D9'],
  mono:     ['#64748b','#1e293b'],
};

/*
 * Status colours for chart annotations (biggest gain / biggest drop markers).
 *
 * Resolved from the theme tokens at call time rather than written as literals:
 * ECharts paints to a canvas and cannot resolve var(), so the value has to be a
 * concrete string by the time it reaches the option object. Both portal pages
 * previously carried their own copy of a Tailwind green/red pair (#15803d /
 * #dc2626) plus eight hand-tuned rgba() variants, which meant the markers were
 * the one part of a chart that did not follow the product's success/danger
 * colours or change with the theme.
 *
 * The literals below are fallbacks for a missing token, not the design.
 */
/*
 * Sequential ramp for the cohort heatmap. Not categorical: the order carries
 * magnitude, so it runs light-to-dark through the brand hue rather than using
 * distinct hues. Kept here with the other chart colours instead of inline in
 * the template, where it was the one array that escaped the shared file.
 */
window.QB_HEATMAP_RAMP = ['#EAF3F0', '#3FC0A9', '#0A6154'];

window.QB_CHART_STATUS = function () {
  const root = getComputedStyle(document.documentElement);
  const token = (name, fallback) => root.getPropertyValue(name).trim() || fallback;

  const channels = (value) => {
    const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      const h = hex[1].length === 3
        ? hex[1].split('').map((c) => c + c).join('')
        : hex[1];
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    const nums = (value.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
    // color(srgb r g b) reports 0-1; rgb() reports 0-255.
    return nums.length === 3
      ? (nums.every((n) => n <= 1) ? nums.map((n) => Math.round(n * 255)) : nums)
      : [0, 0, 0];
  };

  const good = token('--success', '#15803d');
  const bad = token('--danger', '#dc2626');
  const alpha = (value, a) => 'rgba(' + channels(value).join(',') + ',' + a + ')';

  return {
    color:      { gain: good, drop: bad },
    label:      { gain: good, drop: bad },
    background: { gain: alpha(good, 0.09), drop: alpha(bad, 0.09) },
    border:     { gain: alpha(good, 0.25), drop: alpha(bad, 0.25) },
    shadow:     { gain: alpha(good, 0.22), drop: alpha(bad, 0.24) },
    ring:       token('--surface', '#FCFDFC'),
  };
};

/*
 * Chart chrome — axis labels, axis line, gridlines, tooltip.
 *
 * Same reasoning as the status colours: canvas needs concrete strings, so these
 * resolve from the theme tokens at call time. Both portal pages previously
 * carried their own copy of a Tailwind slate set (#94A3B8 / #64748B / #334155 /
 * #CBD5E1 and matching rgba gridlines), which meant every chart's furniture sat
 * in a different neutral family from the product around it — the frame told you
 * it came from somewhere else even when the data was right.
 */
window.QB_CHART_THEME = function () {
  const root = getComputedStyle(document.documentElement);
  const token = (name, fallback) => root.getPropertyValue(name).trim() || fallback;

  const channels = (value) => {
    const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      const h = hex[1].length === 3
        ? hex[1].split('').map((c) => c + c).join('')
        : hex[1];
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    const nums = (value.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
    return nums.length === 3
      ? (nums.every((n) => n <= 1) ? nums.map((n) => Math.round(n * 255)) : nums)
      : [0, 0, 0];
  };
  const alpha = (value, a) => 'rgba(' + channels(value).join(',') + ',' + a + ')';

  const muted = token('--text-muted', '#5E706A');
  const border = token('--border', '#D8E2DD');
  const surface = token('--surface', '#FCFDFC');
  const text = token('--text', '#101C18');

  return {
    // Retained and always false: callers that still pass it through to ECharts'
    // theme argument keep working, and removing the key would break them
    // silently rather than loudly.
    dark: false,
    axis:        muted,
    // Marks that punch a ring out of the page (pie slice borders, treemap
    // gaps) need the surface as a concrete value.
    surface,
    axisLine:    border,
    // Gridlines are structure, not data: present enough to read a value
    // against, quiet enough that the series stays the loudest thing.
    split:       alpha(border, 0.85),
    tooltipBg:   alpha(surface, 0.97),
    tooltipText: text,
  };
};
