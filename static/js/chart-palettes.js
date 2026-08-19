/*
 * Chart series palettes, shared by the portal chat and the portal dashboard.
 *
 * These are deliberately literal hex values, not design tokens: ECharts paints
 * to a canvas and cannot resolve var(), and a categorical series set is a
 * designed qualitative ramp rather than a semantic colour. Each palette carries
 * its own light and dark array because a categorical ramp that reads well on
 * white does not read well on #0f172a.
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
  default: {
    light: ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'],
    dark:  ['#3987e5','#d95926','#199e70','#c98500','#d55181','#008300','#9085e9','#e66767'],
  },
  ocean: {
    light: ['#14B8A6','#1D4ED8','#0EA5E9','#2347B8','#0D9488','#2563EB','#38BDF8','#0369A1'],
    dark:  ['#0E8E7E','#1D4ED8','#1590C4','#2C55C9','#0D9488','#2563EB','#1E9BDB','#0B6598'],
  },
  // Best-achievable, not fully passing -- see comment above.
  sunset: {
    light: ['#F97316','#9A3412','#F59E0B','#DC2626','#EC4899','#C2410C','#A21CAF','#E11D48'],
    dark:  ['#EA580C','#A8441A','#B8860B','#DC2626','#EC4899','#C2410C','#A21CAF','#E11D48'],
  },
  // Best-achievable, not fully passing -- see comment above.
  forest: {
    light: ['#10B981','#00704E','#22C55E','#4D7C0F','#059669','#84CC16','#166534','#2FBF8E'],
    dark:  ['#0D9668','#00704E','#189A6C','#4D7C0F','#059669','#5CA815','#15803D','#0F9D6D'],
  },
  candy: {
    light: ['#EC4899','#CA8A04','#8B5CF6','#14B8A6','#6366F1','#F97316','#0891B2','#DC2626'],
    dark:  ['#EC4899','#A8710A','#8B5CF6','#0F9D6D','#6366F1','#EA580C','#0891B2','#DC2626'],
  },
  // Reclassified as a one-hue ORDINAL ramp, not an 8-slot categorical set --
  // true grayscale has ~0 OKLCH chroma, which fails the categorical chroma
  // floor by design (a hue-based check can't apply to a hue-less palette).
  // Validated instead with the skill's ordinal checks: monotone lightness,
  // >=0.06 ΔL between adjacent steps, light/dark-end contrast against the
  // surface, single-hue spread. Use for genuinely ORDERED series (ranked
  // tiers, funnel stages) where a light-to-dark reading carries the order,
  // not for arbitrary unordered category identity.
  mono: {
    light: ['#a3acb8','#8a95a3','#69748a','#556173','#3d495c','#293344','#151d2b'],
    dark:  ['#3f4b5f','#515e73','#657288','#8a95a3','#a3acb8','#bcc5d0','#d5dce4'],
  },
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
window.QB_CHART_STATUS = function (dark) {
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
    // In dark mode the tokens are already the lighter variants, so the label
    // takes the same colour rather than a separately-tuned tint.
    label:      { gain: good, drop: bad },
    background: { gain: alpha(good, dark ? 0.16 : 0.09),
                  drop: alpha(bad,  dark ? 0.16 : 0.09) },
    border:     { gain: alpha(good, dark ? 0.32 : 0.25),
                  drop: alpha(bad,  dark ? 0.32 : 0.25) },
    shadow:     { gain: alpha(good, 0.22), drop: alpha(bad, 0.24) },
    ring:       token(dark ? '--surface' : '--surface', dark ? '#111827' : '#ffffff'),
  };
};
