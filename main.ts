// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Forecast: for each hour h = 2..24 of launch day, five quantiles of that
// hour's concurrent Twitch viewers in the launch game's category, as ratios
// to hour 1's print p1.
//
//   q_tau(h) = p1 * exp(m_h + z_tau * s_h)
//
// where m_h is the fitted median log-ratio ln(viewers_h / viewers_1) and s_h
// its spread, both learned at run time from the case file's own
// `comparables` (past big launches' hourly curves, 14 days before to 3 days
// after their release day). Nothing is hardcoded from the practice cases:
// the final case carries 42 comparables, the practice cases 38, and the
// program fits whatever it is given.
//
// Why the shape is conditioned on hour 1's size: across the comparables,
// log(viewers_h / viewers_1) correlates about -0.93 with log(viewers_1).
// A launch that is already huge in hour 1 (midnight rolling unlock, most
// regions live) dips to roughly half by midday and recovers to about its
// hour-1 level in the evening; a launch whose hour 1 is still quiet
// (simultaneous global unlock later in the day) multiplies several-fold.
// A single unconditional curve cannot hold both regimes, and persistence
// (the house baseline) holds neither: it stays flat with a spread that
// only widens like sqrt(h-1).
//
// Fit (per hour h, independently, then smoothed across hours):
//   - Big launches (hour-1 print >= 20,000, the regime Grand Theft Auto VI
//     will be in): Theil-Sen regression of the log-ratio on log hour-1
//     print, over the comparables with hour-1 print >= 20,000. Slope
//     shrunk by 0.85 toward 0 (regularisation against 27 noisy points),
//     intercept re-centred on the pool means. Spread s_h is the
//     MAD-based residual scale x 1.15 (the raw MAD scale under-covers).
//     The target's log hour-1 print is clipped to the pool's range, so a
//     record-breaking hour 1 gets the biggest launch's curve, never an
//     extrapolated one.
//   - Small launches (hour-1 print < 20,000): ordinary least squares of
//     the log-ratio on log hour-1 print over all comparables, spread the
//     residual standard deviation. This is the better fit where the
//     log-ratio explodes (up to 2000x) and Theil-Sen's median slope
//     under-fits.
//   - If fewer than 8 comparables have a complete launch day, fall back
//     to the house baseline (persistence, spread 0.15 per sqrt hour).
// Median and spread curves are smoothed with a 3-point moving average so
// the forecast does not zigzag hour to hour.
//
// Backtest (leave-one-out over the 37 comparable launches with complete
// launch days, mean pinball loss on the log scale, exactly the
// competition's scoring rule): big-launch regime 0.250 vs the house
// baseline's 0.321 (-22%); over all 37 launches 0.345 vs 0.930 (-63%,
// the baseline collapses on quiet-hour-1 launches). The four practice
// cases cannot be scored locally: their files stop at the checkpoint and
// hold no launch-day actuals, so the comparables are the validation set.
// See README.md for the full table.
//
// It runs offline, with no network, deterministic (no randomness, inputs
// processed in name order, comparables sorted by name), well under the
// 8-second cap.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** A launch counts as "big" (already live in most regions at hour 1) at this hour-1 print. */
const BIG_H1 = 20000;
/** Shrinkage of the Theil-Sen slope toward 0. */
const SLOPE_SHRINK = 0.85;
/** Inflation of the MAD-based residual scale (it under-covers unscaled). */
const SD_INFLATE = 1.15;
/** The house baseline's spread, used only if the comparables cannot be fitted. */
const BASELINE_SIGMA = 0.15;

interface Comparable {
  name: string;
  day_start: string;
  hourly: Array<{ from: string; viewers: number }>;
}

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/** 3-point moving average; endpoints average themselves twice. */
const smooth = (vals: number[]): number[] =>
  vals.map((v, i) => {
    const lo = i > 0 ? vals[i - 1] : v;
    const hi = i < vals.length - 1 ? vals[i + 1] : v;
    return (lo + v + hi) / 3;
  });

/** Ordinary least squares of ys on xs: [intercept, slope, residual sd]. */
const ols = (xs: number[], ys: number[]): [number, number, number] => {
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  const b = sxx === 0 ? 0 : sxy / sxx;
  const a = my - b * mx;
  const resid = ys.map((y, i) => y - (a + b * xs[i]));
  const sd = Math.sqrt(resid.reduce((s, r) => s + r * r, 0) / Math.max(xs.length - 2, 1));
  return [a, b, sd];
};

/** Theil-Sen fit of ys on xs: [intercept, slope, MAD-based residual scale]. */
const theilSen = (xs: number[], ys: number[]): [number, number, number] => {
  const slopes: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      if (xs[j] !== xs[i]) slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
    }
  }
  const b = slopes.length === 0 ? 0 : median(slopes);
  const a = median(ys.map((y, i) => y - b * xs[i]));
  const resid = ys.map((y, i) => y - (a + b * xs[i]));
  const medResid = median(resid);
  const mad = median(resid.map((r) => Math.abs(r - medResid)));
  const sd = Math.max(1.4826 * mad, 0.05);
  return [a, b, sd];
};

interface LaunchCurve {
  /** log(viewers_h / viewers_1) for h = 1..24. */
  logRatio: number[];
  /** Hour-1 print. */
  h1: number;
}

/** A comparable's launch-day curve, or null when its launch day is incomplete. */
const launchCurve = (c: Comparable): LaunchCurve | null => {
  const dayStart = Date.parse(c.day_start);
  if (Number.isNaN(dayStart)) return null;
  const byHour = new Map<number, number>();
  for (const p of c.hourly ?? []) {
    const t = Date.parse(p.from);
    if (Number.isNaN(t) || typeof p.viewers !== 'number') continue;
    const h = Math.round((t - dayStart) / 3600000);
    if (h >= 0 && h < 24 && !byHour.has(h)) byHour.set(h, p.viewers);
  }
  if (byHour.size !== 24) return null;
  const h1 = byHour.get(0);
  if (h1 === undefined || h1 < 1) return null;
  const logRatio: number[] = [];
  for (let h = 0; h < 24; h++) {
    const v = byHour.get(h);
    if (v === undefined || v < 1) return null;
    logRatio.push(Math.log(v / h1));
  }
  return { logRatio, h1 };
};

/** Median log-ratio curve m_h and spread s_h for h = 1..24, fitted on a pool. */
const fitCurve = (
  pool: LaunchCurve[],
  logP1: number,
  big: boolean,
): { med: number[]; sd: number[] } => {
  const xs = pool.map((c) => Math.log(c.h1));
  const xt = big ? Math.min(Math.max(logP1, Math.min(...xs)), Math.max(...xs)) : logP1;
  const med: number[] = [];
  const sd: number[] = [];
  for (let h = 0; h < 24; h++) {
    const ys = pool.map((c) => c.logRatio[h]);
    if (big) {
      const [, bRaw, rsd] = theilSen(xs, ys);
      const b = bRaw * SLOPE_SHRINK;
      const a = mean(ys) - b * mean(xs);
      med.push(a + b * xt);
      sd.push(rsd * SD_INFLATE);
    } else {
      const [a, b, rsd] = ols(xs, ys);
      med.push(a + b * xt);
      sd.push(rsd);
    }
  }
  return { med: smooth(med), sd: smooth(sd) };
};

/** The five quantiles for one hour, from the fitted log-ratio median and spread. */
const hourQuantiles = (p1: number, m: number, s: number): number[] => {
  const spread = Math.max(s, 0.05);
  const qs = Z.map((z) => round2(p1 * Math.exp(m + z * spread)));
  // Rounding must not break the ordering or the bounds the format requires.
  for (let i = 0; i < qs.length; i++) {
    qs[i] = Math.min(1e9, Math.max(1, qs[i]));
    if (i > 0 && qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
};

/** The house baseline's quantiles, used only when the comparables cannot be fitted. */
const baselineQuantiles = (p1: number, h: number): number[] => {
  const qs = Z.map((z) => round2(p1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1))));
  for (let i = 0; i < qs.length; i++) {
    qs[i] = Math.min(1e9, Math.max(1, qs[i]));
    if (i > 0 && qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  }
  return qs;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);

  const curves = (c.comparables ?? [])
    .slice()
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    .map(launchCurve)
    .filter((x): x is LaunchCurve => x !== null);

  let hours: Array<{ hour: number; q: number[] }>;
  const bigPool = curves.filter((x) => x.h1 >= BIG_H1);
  if (curves.length >= 8 && p1 >= BIG_H1 && bigPool.length >= 12) {
    const { med, sd } = fitCurve(bigPool, Math.log(p1), true);
    hours = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, med[hour - 1], sd[hour - 1]) }));
  } else if (curves.length >= 8) {
    const { med, sd } = fitCurve(curves, Math.log(p1), false);
    hours = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, med[hour - 1], sd[hour - 1]) }));
  } else {
    hours = c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
