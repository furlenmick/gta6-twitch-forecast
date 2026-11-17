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
  in): per-hour **Theil–Sen regression** of the log-ratio on log hour-1
  print, over the comparables with hour-1 print ≥ 20,000. Slope shrunk
  by 0.85 toward 0 (regularisation against ~27 noisy points), intercept
  re-centred on the pool means. Spread is the MAD-based residual scale
  × 1.15 (the raw MAD scale under-covers). The target's log hour-1 print
  is clipped to the pool's range, so a record-breaking hour 1 gets the
  biggest launch's curve, never an extrapolated one.
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

The launch game's own pre-release history was tested as a second signal
(hour 1 ÷ 7-day pre-release mean): its correlation with the peak
log-ratio is only −0.26, and it adds nothing once log hour-1 is in the
model, so it is not used.

## Backtest

Scoring is exactly the competition's rule: mean pinball loss on the log
scale over every scored hour and level. **The four practice cases cannot
be scored locally** — their files stop at the checkpoint and hold no
launch-day actuals (their `history_hourly` ends at hour 1, and their
comparables deliberately leave out all four practice launches). So the
validation set is the comparables themselves: **leave-one-out over the
37 comparable launches with complete launch days** (each target scored
against a model fitted on the other 36; the target's own curve never
leaks into its forecast).

| Set | This entry | House baseline | Improvement |
| --- | --- | --- | --- |
| Big launches, hour-1 ≥ 20k (n=27, the GTA regime) | **0.250** | 0.321 | −22% |
| Small launches, hour-1 < 20k (n=10) | **0.705** | 2.572 | −73% |
| All 37 launches | **0.373** | 0.930 | −60% |

90% coverage on the big-launch set is 0.88 (honest ≈ 0.90; the baseline
manages 0.79). The entry beats the baseline on 21 of the 27 big launches;
its worst big-launch misses are idiosyncratic unlock-timing cases
(Monster Hunter Wilds 0.43 vs baseline 0.43, Resident Evil Requiem 1.11
vs 1.27 — both still beat the baseline). Variants tested and rejected:
unconditional median curve (0.793 overall), conditioning on the
pre-release ratio instead of hour-1 size (0.774), k-nearest neighbours
in log hour-1 (0.258 big), blending the regression median halfway to
persistence (0.276 big).

Sanity check on the practice cases (predictions, not scores — no truth
available): for God of War Ragnarök's hour-1 print (269,517) the model
predicts a midday dip to ~0.7× and an evening peak of ~1.3×; for a
record 875k hour 1 it predicts a dip to ~0.6× and an evening of
~0.9×, matching the ELDEN RING / Cyberpunk 2077 shapes. A 2M hour-1
print gets the same curve as 875k (clipped, no extrapolation).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory, writes `<id>.out` to the
output directory: 23 rows, hours 2–24, five quantiles each. Deterministic
(two runs byte-identical), ~0.2 s per case, no network calls.

## Files

- `main.ts` — the entry (Node 22, standard library only)
- `LICENSE` — MIT
- `README.md` — this file

Data: the case files' comparables are SullyGnome.com hourly Twitch
category averages, quoted with attribution; no bulk download was made.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
