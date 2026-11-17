// gta6-twitch-ccu-forecast-01 entry, v2 (Node 22, standard library only).
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
// v2 vs v1: the big-launch fit is now a two-feature regression instead of
// Theil-Sen on log hour-1 alone. The second feature is the launch's
// rollout state at 12 hours before its release day:
//   lag12 = ln( max( viewers(day_start - 12h) / mean viewers of the 7
//                     pre-release days, 0.01 ) )
// A launch whose category was already surging 12 hours before day_start
// (rolling midnight unlocks reaching the big regions) has a different
// day shape from one that was still quiet then (simultaneous unlock later
// in the day). The feature is computed for every comparable from its own
// hourly points, and for the target from the case's history_hourly, so it
// is available at run time for any case, including the final one. It is
// the one extra signal that earned its keep in leave-one-out testing:
// big-launch loss 0.250 -> 0.239, wins vs the house baseline 20/27 ->
// 24/27, 90% coverage 0.876 -> 0.908. Everything else that was tried and
// did not help is listed in README.md.
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
//     will be in): ordinary least squares of the log-ratio on two
//     features, log hour-1 print and lag12, over the comparables with
//     hour-1 print >= 20,000 and a computable lag12. Slopes shrunk by
//     0.85 toward 0 (regularisation against ~27 noisy points), intercept
//     re-centred on the pool means. Spread s_h is the residual standard
//     deviation x 1.15 (the raw scale under-covers). Each target feature
//     is clipped to the pool's range, so a record-breaking hour 1 gets
//     the biggest launch's curve, never an extrapolated one.
//   - Small launches (hour-1 print < 20,000): ordinary least squares of
//     the log-ratio on log hour-1 print over all comparables, spread the
//     residual standard deviation. This is the better fit where the
//     log-ratio explodes (up to 2000x).
//   - If fewer than 8 comparables have a complete launch day, or the
//     target's lag12 cannot be computed, the small-launch fit is used
//     when possible; if even that is impossible, fall back to the house
//     baseline (persistence, spread 0.15 per sqrt hour).
// Median and spread curves are smoothed with a 3-point moving average so
// the forecast does not zigzag hour to hour.
//
// Backtest (leave-one-out over the 37 comparable launches with complete
// launch days, mean pinball loss on the log scale, exactly the
// competition's scoring rule): big-launch regime 0.239 vs the house
// baseline's 0.321 (-26%); over all 37 launches 0.374 vs 0.930 (-60%,
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
/** Shrinkage of the regression slopes toward 0. */
const SLOPE_SHRINK = 0.85;
/** Inflation of the residual scale (the raw scale under-covers). */
const SD_INFLATE = 1.15;
/** The house baseline's spread, used only if the comparables cannot be fitted. */
const BASELINE_SIGMA = 0.15;
/** The pre-release mean is taken over this many days before day_start. */
const PRE_DAYS = 7;
/** The rollout-state feature is read this many hours before day_start. */
const LAG_HOURS = 12;

interface HourlyPoint {
  from?: string;
  at?: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface Case {
  levels: number[];
  day_start?: string;
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: HourlyPoint[];
  history?: HourlyPoint[];
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/** 3-point moving average; endpoints average themselves twice. */
const smooth = (vals: number[]): number[] =>
  vals.map((v, i) => {
    const lo = i > 0 ? vals[i - 1] : v;
    const hi = i < vals.length - 1 ? vals[i + 1] : v;
    return (lo + v + hi) / 3;
  });

/**
 * Ordinary least squares of ys on k features: returns [intercept, slopes,
 * residual sd]. Solves the normal equations by Gaussian elimination with
 * partial pivoting; deterministic for a given input order.
 */
const olsMulti = (xs: number[][], ys: number[]): [number, number[], number] => {
  const n = ys.length;
  const k = xs.length + 1;
  // Design rows: [1, x1, x2, ...].
  const a: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = [1];
    for (const col of xs) row.push(col[i]);
    a.push(row);
  }
  // Normal equations AtA * coef = Atb.
  const m: number[][] = [];
  for (let i = 0; i < k; i++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += a[r][i] * a[r][j];
      row.push(s);
    }
    let b = 0;
    for (let r = 0; r < n; r++) b += a[r][i] * ys[r];
    row.push(b);
    m.push(row);
  }
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let r = col + 1; r < k; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    }
    [m[col], m[piv]] = [m[piv], m[col]];
    if (Math.abs(m[col][col]) < 1e-12) continue;
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let cc = col; cc <= k; cc++) m[r][cc] -= f * m[col][cc];
    }
  }
  const coef = m.map((row, i) => (Math.abs(row[i]) > 1e-12 ? row[k] / row[i] : 0));
  let ssr = 0;
  for (let i = 0; i < n; i++) {
    let pred = coef[0];
    for (let j = 0; j < xs.length; j++) pred += coef[j + 1] * xs[j][i];
    ssr += (ys[i] - pred) ** 2;
  }
  const sd = Math.sqrt(ssr / Math.max(n - k, 1));
  return [coef[0], coef.slice(1), Math.max(sd, 0.05)];
};

/** Ordinary least squares of ys on one feature: [intercept, slope, residual sd]. */
const ols = (xs: number[], ys: number[]): [number, number, number] => {
  const [a, b, sd] = olsMulti([xs], ys);
  return [a, b[0], sd];
};

interface LaunchCurve {
  /** log(viewers_h / viewers_1) for h = 1..24. */
  logRatio: number[];
  /** Hour-1 print. */
  h1: number;
  /** Rollout-state feature, or null when the pre-release points are missing. */
  lag12: number | null;
}

/**
 * The rollout-state feature for a launch whose day starts at `dayStartMs`
 * and whose pre-release points are `points` (each {time, viewers}):
 *   ln( max( viewers at the last point at or before dayStart - 12h
 *            / mean viewers over the 7 days before dayStart, 0.01 ) )
 * Returns null when there are no usable pre-release points.
 */
const lag12Feature = (
  dayStartMs: number,
  points: Array<{ t: number; v: number }>,
): number | null => {
  const pre = points.filter((p) => p.t >= dayStartMs - PRE_DAYS * 86400000 && p.t < dayStartMs);
  if (pre.length === 0) return null;
  const pre7 = mean(pre.map((p) => p.v));
  if (!(pre7 > 0)) return null;
  const atLag = dayStartMs - LAG_HOURS * 3600000;
  let lag: number | null = null;
  for (const p of pre) {
    if (p.t <= atLag) lag = p.v;
  }
  if (lag === null || !(lag > 0)) return null;
  return Math.log(Math.max(lag / pre7, 0.01));
};

/** Normalise a case/comparable's hourly points to {t, viewers} pairs. */
const pointsOf = (pts: HourlyPoint[] | undefined): Array<{ t: number; v: number }> => {
  const out: Array<{ t: number; v: number }> = [];
  for (const p of pts ?? []) {
    const t = Date.parse(p.from ?? p.at ?? '');
    if (!Number.isNaN(t) && typeof p.viewers === 'number' && p.viewers > 0) {
      out.push({ t, v: p.viewers });
    }
  }
  out.sort((a, b) => a.t - b.t);
  return out;
};

/** A comparable's launch-day curve, or null when its launch day is incomplete. */
const launchCurve = (c: Comparable): LaunchCurve | null => {
  const dayStart = Date.parse(c.day_start);
  if (Number.isNaN(dayStart)) return null;
  const byHour = new Map<number, number>();
  for (const p of c.hourly ?? []) {
    const t = Date.parse(p.from ?? '');
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
  const lag12 = lag12Feature(dayStart, pointsOf(c.hourly));
  return { logRatio, h1, lag12 };
};

/** Median log-ratio curve m_h and spread s_h for h = 1..24, big-launch fit. */
const fitCurveBig = (
  pool: LaunchCurve[],
  logP1: number,
  lagFeat: number,
): { med: number[]; sd: number[] } => {
  const xs1 = pool.map((c) => Math.log(c.h1));
  const xs2 = pool.map((c) => c.lag12 as number);
  const xt1 = Math.min(Math.max(logP1, Math.min(...xs1)), Math.max(...xs1));
  const xt2 = Math.min(Math.max(lagFeat, Math.min(...xs2)), Math.max(...xs2));
  const med: number[] = [];
  const sd: number[] = [];
  for (let h = 0; h < 24; h++) {
    const ys = pool.map((c) => c.logRatio[h]);
    const [, slopes, rsd] = olsMulti([xs1, xs2], ys);
    const b1 = slopes[0] * SLOPE_SHRINK;
    const b2 = slopes[1] * SLOPE_SHRINK;
    const a = mean(ys) - b1 * mean(xs1) - b2 * mean(xs2);
    med.push(a + b1 * xt1 + b2 * xt2);
    sd.push(rsd * SD_INFLATE);
  }
  return { med: smooth(med), sd: smooth(sd) };
};

/** Median log-ratio curve m_h and spread s_h for h = 1..24, small-launch fit. */
const fitCurveSmall = (
  pool: LaunchCurve[],
  logP1: number,
): { med: number[]; sd: number[] } => {
  const xs = pool.map((c) => Math.log(c.h1));
  const xt = Math.min(Math.max(logP1, Math.min(...xs)), Math.max(...xs));
  const med: number[] = [];
  const sd: number[] = [];
  for (let h = 0; h < 24; h++) {
    const ys = pool.map((c) => c.logRatio[h]);
    const [a, b, rsd] = ols(xs, ys);
    med.push(a + b * xt);
    sd.push(rsd);
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

  // The target's own rollout-state feature, from its pre-release history.
  let targetLag: number | null = null;
  if (c.day_start !== undefined) {
    const ds = Date.parse(c.day_start);
    if (!Number.isNaN(ds)) {
      const pts = pointsOf(c.history_hourly);
      targetLag = lag12Feature(ds, pts.length > 0 ? pts : pointsOf(c.history));
    }
  }

  let hours: Array<{ hour: number; q: number[] }>;
  const bigPool = curves.filter((x) => x.h1 >= BIG_H1 && x.lag12 !== null);
  if (curves.length >= 8 && p1 >= BIG_H1 && bigPool.length >= 12 && targetLag !== null) {
    const { med, sd } = fitCurveBig(bigPool, Math.log(p1), targetLag);
    hours = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, med[hour - 1], sd[hour - 1]) }));
  } else if (curves.length >= 8) {
    const { med, sd } = fitCurveSmall(curves, Math.log(p1));
    hours = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, med[hour - 1], sd[hour - 1]) }));
  } else {
    hours = c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
