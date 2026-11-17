# GTA VI Twitch CCU forecast — make.fun `gta6-twitch-ccu-forecast-01`

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dhamskie-ai/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dhamskie-ai/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entered in the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) "Forecast GTA VI's launch-day Twitch viewers hour by hour" ($250 in USDC).

A program that forecasts, hour by hour, how many people watch Grand Theft
Auto VI live on Twitch over the 24 UTC hours of its release day
(2026-11-19), from everything recorded up to the end of its first hour,
as five quantiles (5%, 25%, 50%, 75%, 95%) per hour.

## Run

```
python3 main.py            # reads $MAKEFUN_INPUT_DIR/*.in, writes $MAKEFUN_OUTPUT_DIR/*.out
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library
only, no network) with one case file in `/input`. Locally:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Model

A comparable-launch shape model, anchored to the case's hour-1 print.

For every past big launch in the case's `comparables`, take its release-day
hourly curve and form log-ratios `r_h = ln(viewers_h / viewers_1)`. The
median of `r_h` for the target is estimated two ways and blended
(0.6 / 0.4):

- **kNN in log(hour-1):** Gaussian kernel over `|ln(h1) - ln(p1)|`
  (bandwidth 1.5), weighted by unlock class — rolling-midnight launches count
  most (that is what GTA VI is), global-simultaneous and early-access
  launches less. This captures the dominant fact in the data: a launch whose
  hour 1 is already huge (most of the world unlocked) barely grows, while
  one whose hour 1 is tiny (unlock still coming) grows ten- to a hundred-fold.
  Across the 37 past launches the correlation of `ln(h1)` with `r_h` reaches
  −0.93 by hour 19.
- **OLS** of `r_h` on `ln(h1)`, clamped to the observed range, which
  smooths the kNN estimate where neighbours are sparse.

The spread is split: the weighted standard deviation of the neighbours'
residuals above the median sets the upper spread, below it the lower spread
(the upside tail is fatter — some launches explode), each floored at 0.20 log
units. Quantiles are `q = p1 * exp(median + z_tau * spread_side)`.

If a case has no usable comparable, the program falls back to the house
persistence baseline (median = hour-1 print, spread 0.15·√(h−1)).

## Validation (local, log-scale pinball loss)

Leave-one-launch-out over the 37 past launches in the kit, plus the 4
practice launches scored against their SullyGnome hourly counts:

| model | practice mean (4 cases) | 37-launch LOLO mean |
|---|---|---|
| house persistence baseline | 0.387 | 0.930 |
| this model | **0.179** | **0.414** |

Practice cases individually: Wolverine 0.209, Red Dead Redemption 2 0.176,
God of War Ragnarök 0.139, Marvel's Spider-Man 2 0.193. 90% coverage on
practice: 0.94. Runtime per case: well under 1 second; output is
deterministic (two runs on the same case are byte-identical).

## Files

- `main.py` — the whole entry (standard library only).
- `LICENSE` — MIT, copied from the kit starter.
