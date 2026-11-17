[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/furlenmick/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/furlenmick/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — make.fun entry

Forecasts, hour by hour, how many concurrent viewers the Grand Theft Auto VI
Twitch category will have on release day (2026-11-19 UTC), hours 2–24, as
five quantiles (5/25/50/75/95%) of each hour's count.

## How it works

For each hour `h`, the forecast is a ratio to hour 1's print `p1`:

```
q_tau(h) = p1 * exp(m_h + z_tau * s_h)
```

`m_h` is the fitted median of `ln(viewers_h / viewers_1)` and `s_h` its
spread, both fitted **at run time from the case file's own `comparables`**
— past big launches' hourly Twitch curves (14 days before to 3 days after
their release day). Nothing is hardcoded from the practice cases: the final
case carries 42 comparables, the practice cases 38, and the program fits
whatever it is given. No network, no dependencies, Node 22 standard
library only.

### The one thing that matters: the shape depends on how big hour 1 is

Across the comparables, `ln(viewers_h / viewers_1)` correlates about
**−0.93** with `ln(viewers_1)`. Two regimes:

- **Hour 1 already huge** (rolling midnight unlock, most regions live):
  the count dips to roughly half by midday, then recovers to about its
  hour-1 level in the evening (e.g. ELDEN RING: hour 1 875k, evening
  ~0.9×; Cyberpunk 2077: 852k, evening ~0.9×).
- **Hour 1 still quiet** (simultaneous global unlock later in the day):
  the count multiplies several-fold (e.g. Hollow Knight: Silksong:
  hour 1 164 viewers, peak 2,121×; Battlefield 6: 2,470 → 228×).

A single unconditional curve cannot hold both regimes, and persistence
(the house baseline — flat at hour 1, spread widening like `sqrt(h−1)`) —
holds neither. So the fit is conditioned on hour 1's size:

- **Big launches** (hour-1 print ≥ 20,000 — the regime GTA VI will be
  in): per-hour **ordinary least squares** of the log-ratio on two
  features — log hour-1 print, and the launch's **rollout state 12 hours
  before its release day** (`lag12`: the log of the category's viewers at
  day_start−12h divided by its mean over the 7 pre-release days, floored
  at 0.01). A launch whose category was already surging 12 hours before
  day_start (rolling midnight unlocks reaching the big regions) has a
  different day shape from one that was still quiet then (simultaneous
  unlock later in the day). The feature is computed for every comparable
  from its own hourly points and for the target from the case's
  `history_hourly`, so it is available at run time for any case. Slopes
  shrunk by 0.85 toward 0 (regularisation against ~27 noisy points),
  intercept re-centred on the pool means; spread is the residual standard
  deviation × 1.15 (the raw scale under-covers). Each target feature is
  clipped to the pool's range, so a record-breaking hour 1 gets the
  biggest launch's curve, never an extrapolated one. (v1 used Theil–Sen
  on log hour-1 alone; the second feature is the only extra signal that
  earned its keep in leave-one-out testing — see below.)
- **Small launches** (hour-1 print < 20,000): per-hour **ordinary least
  squares** of the log-ratio on log hour-1 print over all comparables,
  spread the residual standard deviation. This fits better where the
  log-ratio explodes (up to 2000×) and Theil–Sen's median slope
  under-fits.
- **Fallback:** if fewer than 8 comparables have a complete launch day,
  the house baseline (persistence, spread 0.15 per √hour).

Median and spread curves are smoothed with a 3-point moving average so
the forecast does not zigzag hour to hour. Quantiles are rounded to 2
decimals, forced non-decreasing, clamped to [1, 1e9].

## Backtest

Scoring is exactly the competition's rule: mean pinball loss on the log
scale over every scored hour and level. **The four practice cases cannot
be scored locally** — their files stop at the checkpoint and hold no
launch-day actuals (their `history_hourly` ends at hour 1, and their
comparables deliberately leave out all four practice launches). So the
validation set is the comparables themselves: **leave-one-out over the
37 comparable launches with complete launch days** (each target scored
against a model fitted on the other 36; the target's own curve never
leaks into its forecast). Numbers below are from running the shipped
`main.ts` end to end on those 37 leave-one-out cases, so they are the
numbers this exact code produces.

| Set | This entry (v2) | v1 | House baseline | Improvement vs baseline |
| --- | --- | --- | --- | --- |
| Big launches, hour-1 ≥ 20k (n=27, the GTA regime) | **0.239** | 0.250 | 0.321 | −26% |
| Small launches, hour-1 < 20k (n=10) | **0.738** | 0.738 | 2.572 | −71% |
| All 37 launches | **0.374** | 0.382 | 0.930 | −60% |

(The small-launch fit is unchanged from v1, so its loss is identical;
the v1 README's 0.705 / 0.373 came from its Python backtest harness,
whose small-regime fit differed slightly from the shipped code. Under
one protocol — the shipped entry, run end to end — v1 is 0.250 / 0.382
and v2 is 0.239 / 0.374.)

90% coverage on the big-launch set is 0.908 (honest ≈ 0.90; the baseline
manages 0.79; v1 managed 0.876). The entry beats the baseline on 24 of
the 27 big launches (v1: 20/27) and on 33 of all 37. Its worst
big-launch misses are idiosyncratic unlock-timing cases (Resident Evil
Requiem 0.98 vs baseline 1.27, Halo Infinite 0.57 vs 0.60, Borderlands 4
0.36 vs 0.26 — the one case where v2 loses to the baseline by a wide
margin). The `lag12` feature helps the most on the hard cases: Resident
Evil Requiem 1.11 → 0.98, Monster Hunter Wilds 0.43 → 0.30, Starfield
0.34 → 0.23; it costs a little on a few clean giants (Cyberpunk 2077
0.14 → 0.22, Diablo IV 0.15 → 0.21).

### What was tried for v2 and rejected

All numbers are leave-one-out mean pinball loss on the big-launch set
(v1: 0.250), tested in Python against the same protocol before anything
was written into `main.ts`:

- **Unlock alignment by real unlock time** (re-express each comparable
  in hours-since-unlock, fit there, map back): the unlock times in the
  case files are mostly free text and often "unverified", and a
  text-based rolling/simultaneous archetype classifier is unreliable
  (it mislabels ELDEN RING and Zelda). Archetype pooling alone: rolling
  pool 0.267, rolling+comparable-tier 0.274 — no gain.
- **Kernel weighting / local-linear fits** on log hour-1 (Gaussian
  kernel, bandwidths 0.8–1.2, rolling pool): 0.258–0.284 — no gain.
- **Scaled-template model** (median archetype curve × a scale predicted
  from log hour-1): 0.278–0.321 — worse; per-hour regressions beat a
  shared shape.
- **Pre-release trajectory matching** (k-nearest neighbours by the
  14-day pre-release daily profile or the 13-hour pre-release ramp,
  k = 1–5, and blends with the regression): 0.30–0.45 alone, blends
  0.250–0.274 — the pre-release trajectory does not predict the launch
  day's shape.
- **Extra features** in the regression: hour-1 channels (0.255), viewers
  per channel (0.253), pre-release ratio R (0.259), pre-release hype
  slope/level (0.255), all four together (0.250) — none beats log
  hour-1 + lag12 (0.239). The day-before hour-of-day profile helps a
  little on its own (0.244) but adds noise on top of lag12 (0.243).
- **Per-hour per-quantile calibration** on leave-one-out residuals
  (empirical residual quantiles blended with the normal spread): 0.244 —
  no gain; the ×1.15 spread inflation already covers honestly.
- **Ensemble** of v1's Theil–Sen median with the two-feature median
  (50/50 in log space): 0.241 big, 0.168 console-only — about tied with
  the two-feature model, so the simpler single model was kept.
- **Console-only pools** (fit only on console-only comparables): the
  pool is too small (7 launches) and the fit degrades (0.28–0.32).

The remaining spread is structural: the per-hour residual standard
deviation of the log-ratio is ~0.5 log-units at later hours and no
pre-checkpoint feature explains more than ~45% of its variance (log
hour-1 alone explains 27–45% at hours 18–24). That is the honest ceiling
of this data: getting from 0.24 to the dev leaderboard's top-256 line
(~0.149) would need information the case files do not contain. The
practice cases are all console-only rolling launches, a subset on which
v1 already scores better (leave-one-out 0.172 over the 7 console-only
big comparables); v2 is 0.174 on that subset, 0.168 as the ensemble
variant — the difference is noise, so the two-feature model was kept
for its big-regime win, which is the regime the final day is in.

Sanity check on the practice cases (predictions, not scores — no truth
available): for God of War Ragnarök's hour-1 print (269,517) the model
predicts a shallow midday dip (~0.9×) and an evening peak of ~1.3×; for
a record 875k hour 1 it predicts the ELDEN RING shape (dip ~0.6×,
evening ~0.9×), clipped, no extrapolation.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory, writes `<id>.out` to the
output directory: 23 rows, hours 2–24, five quantiles each. Deterministic
(two runs byte-identical), ~0.03 s per case (37 cases in 0.9 s), no
network calls.

## Files

- `main.ts` — the entry (Node 22, standard library only)
- `LICENSE` — MIT
- `README.md` — this file

Data: the case files' comparables are SullyGnome.com hourly Twitch
category averages, quoted with attribution; no bulk download was made.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
