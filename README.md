# NBA dynamic strength of schedule (D-SOS)

[![CI](https://github.com/tianrunliao/nba-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/tianrunliao/nba-analytics/actions/workflows/ci.yml)

A pipeline built on official NBA data for the 2007-08 to 2025-26 regular seasons (2006-07 is
used only for the first pre-season rating). It estimates team strength game by game, predicts
single games, models player availability, simulates the rest of a season and measures
remaining schedule difficulty (D-SOS). It also includes playoff and player-value analyses.
Every forecast uses only information available before tip-off, or before the cutoff date.

| Part | What it does | Main code |
|---|---|---|
| Data | ingestion with a raw-file manifest, validation, pre-game features (possessions, efficiency, schedule density, travel, back-to-back context, injury reports), static SOS, pre-season baseline | `ingestion/`, `preprocessing/`, `validation/`, `features/`, `sos/static.py` |
| Strength and game model | Elo, EWMA, dynamic opponent-adjusted strength, Kalman-filter strength, CatBoost single-game model; rolling backtest and feature-group ablation | `ratings/elo.py`, `ratings/dynamic.py`, `ratings/kalman.py`, `models/game_model.py` |
| Availability | probability of playing, missing lineup value, double-counting check, return-time tables | `models/availability.py` |
| Season simulation | average-team counterfactual SOS with a Shapley split, joint league Monte Carlo (strength uncertainty, roster prior, future injuries, official tie-breakers, NBA Cup), historical date replays | `sos/projection.py`, `sos/availability.py`, `sos/roster_prior.py`, `sos/standings.py`, `sos/cup.py`, `models/season_simulation.py` |
| Analysis | player values (box-score prior plus five-man lineup RAPM), playoff strength and its improvements, players who rise in the playoffs, stars vs depth | `ratings/` |
| Live | on-demand snapshot, latest official injury report and projection; Streamlit page | `live.py`, `app/streamlit_app.py` |

## Method

**Pre-game information only.** Result-based features use games on earlier ET dates. Injury
features use the last official report published (HTTP `Last-Modified`) before tip-off.
Strengths at a cutoff use games before the cutoff. Tests check this for every rating
(`tests/test_strength_models.py`, `tests/test_kalman.py`, `tests/test_season_simulation.py`).

**Model selection by whole seasons.** Elo, dynamic-strength and Kalman hyper-parameters are
tuned on 2013-14 to 2015-16, which are never test seasons. CatBoost early-stops on the last
training season. Season-simulation choices (load features, roster prior, availability terms)
use season-grouped k-fold error inside the training seasons. Each test season is only replayed
in time order.

**Kalman strength.** `ratings/kalman.py` treats every team's strength per 100 possessions as
a state: a random walk within a season, regression toward the mean between seasons, and one
joint update per game date. Games identify only strength differences, so the noise is projected
onto sum-to-zero strengths. The grid search over q, p0 and carry uses the dynamic model's
squared-error criterion. The observation sd is then set so that the predictive variance matches
the squared innovations on the tuning seasons. The tuned values are in
`reports/game_model_params.json` (`tuned.kalman`).

## Data

| Source | Content | Location |
|---|---|---|
| stats.nba.com (nba_api 1.12.0) | game logs, team advanced stats (possessions), player game logs, schedules, standings | `data/raw/nba_stats/` |
| official NBA injury report PDFs | pre-game snapshots; HTTP `Last-Modified` is the publication time | `data/raw/injury_reports/` |
| Wikidata P625 | arena coordinates (QIDs checked by hand) | `configs/venue_aliases.csv`, `configs/venues.csv` |

`data/raw/manifest.jsonl` records the URL, fetch time and sha256 of every raw file. The data
come from stats.nba.com and NBA.com, so the NBA terms of use apply, and the data are for
personal research only. Neither raw nor processed data are in this repository (`data/` is about
660 MB, and the NBA terms restrict redistribution). Rebuild them locally with `fetch` and
`build`. The test fixtures that hold real NBA files are not published either. Tests that need
them are skipped; `tests/fixtures/README.md` explains how to recreate them. `reports/` holds
derived results only.

## Results

All numbers below are read from the named file in `reports/`, rounded to the shown precision.

### Single-game prediction

`game-model`, test seasons 2016-17 to 2025-26 (11,979 games), pooled
(`reports/game_model_summary_all.csv`; 2022-23 onward in `game_model_summary_2022_26.csv`).
The target is the home margin per 100 possessions, and win probability is Phi(margin / sigma).

| Model | Log loss | Brier | MAE per 100 | ECE |
|---|---|---|---|---|
| pre-season rating | 0.6558 | 0.2319 | 11.25 | 0.0313 |
| Elo | 0.6250 | 0.2177 | | 0.0428 |
| EWMA (14 days) | 0.6412 | 0.2247 | 10.99 | 0.0306 |
| dynamic strength (linear) | 0.6230 | 0.2169 | 10.68 | 0.0289 |
| Kalman strength (linear) | 0.6217 | 0.2163 | 10.65 | 0.0328 |
| CatBoost: strength | 0.6241 | 0.2172 | 10.68 | 0.0322 |
| CatBoost: strength + Kalman | 0.6236 | 0.2171 | 10.67 | 0.0307 |
| CatBoost: + rest | 0.6224 | 0.2166 | 10.65 | 0.0345 |
| CatBoost: + rest + travel | 0.6225 | 0.2166 | 10.64 | 0.0351 |
| CatBoost: + injury (full) | 0.6177 | 0.2145 | 10.56 | 0.0337 |
| CatBoost full, Platt-calibrated | 0.6176 | 0.2145 | 10.56 | 0.0338 |

Rest and travel features add little: log loss goes from 0.6241 to 0.6224 with rest and to
0.6225 with rest and travel. The injury features, available from 2019-20 on, account for most
of the CatBoost gain (0.6177). CatBoost is less well calibrated than the linear dynamic model
(ECE 0.0322 to 0.0351 against 0.0289). Platt calibration fitted on the held-out last training
season leaves ECE at 0.0338. The Kalman strength has a lower log loss and MAE than the dynamic
strength, but it is less well calibrated (ECE 0.0328 against 0.0289).

### Remaining-season simulation

`replay`: each test season from 2016-17 to 2025-26 is replayed from opening night and from the
dates by which the median team has played 20, 40 and 60 games, giving 300 team-seasons per
cutoff (`reports/season_sim_backtest.csv`). The table shows the mean absolute error of final
wins and the share of teams whose final wins fall inside the 90% interval.

| Games played at cutoff | full: MAE | full: 90% cover | Kalman: MAE | Kalman: 90% cover | first version (v1): MAE | current win% (pace): MAE |
|---|---|---|---|---|---|---|
| 0 | 7.53 | 0.907 | 7.83 | 0.920 | 7.81 | 9.72 |
| 20 | 4.99 | 0.910 | 4.98 | 0.923 | 5.13 | 6.14 |
| 40 | 3.65 | 0.857 | 3.61 | 0.890 | 3.73 | 3.97 |
| 60 | 2.13 | 0.893 | 2.13 | 0.917 | 2.14 | 2.31 |

`full` uses the dynamic strength with a roster prior; `kalman` swaps in the Kalman strengths
and sds without a roster prior, which is why it trails on opening night. Without strength
uncertainty (`fixed_strength`), the 90% interval covers only 0.543 of teams on opening night.
Does the remaining schedule matter beyond a team's own strength? After at least 20 games,
regressing remaining wins on expected wins and on minus D-SOS gives a coefficient of 0.82 (se
0.25) on minus D-SOS (n = 900). Adding D-SOS lowers the MAE of remaining wins from 3.65 to
3.59 (`reports/season_sim_params.json`, `sos_check`).

Pre-season betting lines: `line-check` compares opening-night projections with pre-season win
lines on the same team-seasons, once `configs/preseason_win_totals.csv` has rows. The file is
transcribed by hand from the Basketball-Reference "Preseason Odds" pages (data courtesy of
sportsoddshistory.com), one row per team and season (`season`, `team` as the NBA tricode, e.g.
BKN/CHA/PHX, `line`, `source_url`). 2019-20 is left out (suspended mid-season; lines were set
for 82 games, teams played 64 to 75), and 2020-21 is compared as played (72 games). The file
currently has no rows, so these results are not yet compared with pre-season betting lines.

### Playoffs

`playoff-rank`, postseasons 2011-12 to 2025-26 (1,258 games, 225 series,
`reports/playoff_backtest.csv`). The models are pooled by games for log loss and by series
for the series metrics.

| Model | Log loss | Series accuracy | Series Brier |
|---|---|---|---|
| full (regular-season strength, momentum, playoff-start health) | 0.6343 | 0.720 | 0.1887 |
| regular-season strength only (`theta_only`) | 0.6339 | 0.711 | 0.1868 |
| raw net rating | 0.6361 | 0.693 | 0.1905 |
| win percentage | 0.6346 | 0.711 | 0.1817 |
| `full_injury_adjusted` (after-the-fact reference) | 0.6350 | | |

The full model does not beat strength alone on log loss. `full_injury_adjusted` uses each
game's actual absences, so it is an after-the-fact reference, not a forecast.

Seasons before 2018-19 have no pre-playoff injury report. For these seasons, playoff-start
health comes from a pre-playoff proxy. A rotation player (at least 20 regular-season minutes
per game) who missed the last k consecutive regular-season games counts as out. k = 5
(F1 0.579) is chosen against the Out/Doubtful statuses of teams with a report
(`reports/playoff_model_params.json`, `rs_end_streak_rule`).

`playoff-improve` adds candidate feature groups chosen by nested, time-ordered validation. In
every prediction, the in-playoff absences are set to zero. Playoff form going into a game
lowers pooled log loss from 0.6339 to 0.6306 (`reports/playoff_improve_backtest.csv`). At the
series level, it lowers Brier from 0.1869 to 0.1850 (`reports/playoff_improve_series.csv`).
The player-value variants in `reports/model_variants.csv` do not help: on/off values give
0.6343, lineup RAPM values 0.6370, and adding star/depth terms 0.6378.

### Availability

`availability`, 2021-22 to 2025-26 (6,150 games, `reports/availability_injury_evaluation.csv`,
pooled log loss). Starting from 0.6204 for dynamic strength alone, the official-report counts
bring log loss to 0.6088, and the change in missing lineup value brings it to 0.6071. The
player availability model beats a status-frequency table in every season. In 2025-26, for
example, its log loss is 0.236 against 0.327 (`reports/availability_model.csv`).

### Official injury report

`report-check`, 2020-21 to 2025-26 (`reports/injury_report_check.csv`). At weekly cutoffs, moved
to the next game day, the full model is scored with participation-based availability and again
with the official pre-game report of that day. The report sets each covered rotation player's
probability of missing the next game from the status frequencies of earlier report seasons
(`reports/injury_report_status_miss.json`; for 2025-26: Out 1.000, Doubtful 0.976,
Questionable 0.400, Probable 0.049, not listed 0.033, Available 0.013). The report does not
improve the forecasts overall: on the cutoff day's 912 games, log loss is 0.5962 without it and
0.5965 with it (difference 0.0004, se 0.0027), and on each team's next 10 games (21,907) 0.6199
against 0.6198. On the 150 cutoff-day games whose home-win probability the report moved by more
than 5 points, log loss falls from 0.6380 to 0.6163 (se of the difference 0.0138).

### Current season

`project`, 2026-27 from snapshot `20261123T080609Z` (cutoff 2026-11-23, 248 games played, 952
remaining, 10,000 simulated seasons; `reports/season_sim_projection_2026-27.csv` and `.json`).
San Antonio has the highest expected total, 58.9 wins (90% interval 48 to 68).

## Live projection and Streamlit

```bash
uv run python -m nba_sos.pipeline live
uv run streamlit run app/streamlit_app.py
```

`live` takes a fresh current-season snapshot and fetches the latest official injury report
(at least 30 minutes old, walking back up to one day). It then runs the `project` simulation
and writes `projection.csv`, `injury_report.csv` and `meta.json` to
`reports/live/<snapshot UTC stamp>/` (not tracked by git). The Streamlit page runs the same
flow once when it is opened or refreshed, and again on "Fetch again". It shows the 30-team
table and a per-team view with that team's report rows. Nothing runs on a schedule. The
projection's availability terms come from game participation before the cutoff; for every
team whose next game the official report covers, that game also uses the report statuses (see
"Official injury report" above; `meta.json` records `injury_report_in_projection`).

## Reproduce

```bash
uv sync
uv run python -m nba_sos.pipeline fetch            # network: nba_api payloads + current schedule snapshot
uv run python -m nba_sos.pipeline fetch-injuries   # network: official injury report PDFs (about 2-3 hours)
uv run python -m nba_sos.pipeline build            # offline: raw -> processed tables, features, SOS, baseline, audit
uv run python -m nba_sos.pipeline snapshot         # network: timestamped current-season schedule and results
uv run python -m nba_sos.pipeline window-sos --n 15
uv run python -m nba_sos.pipeline playoff-rank
uv run python -m nba_sos.pipeline analysis
uv run python -m nba_sos.pipeline game-model
uv run python -m nba_sos.pipeline availability
uv run python -m nba_sos.pipeline playoff-improve
uv run python -m nba_sos.pipeline replay
uv run python -m nba_sos.pipeline project
uv run python -m nba_sos.pipeline playoff-risers
uv run python -m nba_sos.pipeline report-check
uv run python -m nba_sos.pipeline line-check       # needs rows in configs/preseason_win_totals.csv
```

`build` reads only `data/raw/` and can be rerun. If an error-level validation check fails, it
writes the audit report and exits with a non-zero status. `analysis` must run before
`playoff-improve` (player values), `game-model` before `availability`, `replay`, `project`
and `report-check` (features and tuned parameters), and `replay` before `line-check`.

| Command | Files written to `reports/` |
|---|---|
| `build` | `data_audit.md`, `static_sos.csv`, `baseline_backtest.csv`, `validation_results.csv` |
| `window-sos --n 15` | `window_sos_<season>_first15.csv`, `window_sos_<season>_first15.params.json` |
| `playoff-rank` | `playoff_backtest.csv`, `playoff_model_params.json`, `playoff_strength_<season>.csv` |
| `analysis` | `player_values_meta.json`, `model_variants.csv`, `star_vs_depth.json`, `top_players_2019_2026.csv`, `team_season_rankings_2013_2026.csv` |
| `game-model` | `game_model_backtest.csv`, `game_model_summary_all.csv`, `game_model_summary_2022_26.csv`, `game_model_params.json` |
| `availability` | `availability_model.csv`, `availability_injury_evaluation.csv`, `availability_return_time.csv`, `availability_status_by_reason.csv`, `availability_params.json` |
| `playoff-improve` | `playoff_improve_backtest.csv`, `playoff_improve_params.json`, `playoff_improve_series.csv` |
| `playoff-risers` | `playoff_risers_career.csv`, `playoff_risers_seasons.csv`, `playoff_risers_params.json` |
| `replay` | `season_sim_replays.csv`, `season_sim_backtest.csv`, `season_sim_params.json` |
| `project` | `season_sim_projection_<season>.csv`, `season_sim_projection_<season>.json` |
| `report-check` | `injury_report_check.csv`, `injury_report_status_miss.json` |
| `line-check` | `season_sim_preseason_lines.csv` (only once the lines file has rows) |
| `live` | `live/<stamp>/projection.csv`, `live/<stamp>/injury_report.csv`, `live/<stamp>/meta.json` (git-ignored) |

The column labels `cb_phase2`, `cb_phase3` and `dyn+phase2_injury` in the availability reports
mean the game-model CatBoost feature set, that set plus lineup features, and dynamic strength
plus the report counts.

## Development

```bash
make install     # uv sync --frozen
make lint        # ruff check + ruff format --check (src, tests, app)
make format
make typecheck   # mypy on src/nba_sos
make test        # pytest (the end-to-end build test needs local data and is skipped without it)
make check       # lint + typecheck + test
uvx pre-commit install
```

CI runs lint, type check and tests on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

## Definitions

| Item | Definition |
|---|---|
| Season scope | regular season only (game ID prefix 002), including NBA Cup group games and the knockout games that count in the standings; the Cup final (006), play-in and playoffs are excluded |
| Possessions | mean of the two teams' POSS; efficiency = points / possessions x 100 |
| Home and away | the schedule's home team ID (at neutral sites both MATCHUP strings use "@") |
| Neutral site | the source `isNeutral` flag or a venue marked neutral (international games, Las Vegas); home-court advantage is 0 there |
| Information cutoff | result features use games on ET dates strictly before the game day; injuries use the last report published before tip-off |
| First-game travel | from the team's home arena |
| Back-to-back context | `b2b_type` (home/away of the previous and current game, e.g. AA = two road games; neutral counts as road), `turnaround_hours`, `prev_overtime_periods`, local tip-off times and the tip-off time in the team's home time zone (body clock), together with `travel_distance_km` and `timezone_change_hours` (positive = eastward) |
| Missing values | season-to-date values before a team's first game and injury features without a usable snapshot stay empty, not imputed |
| Pre-season rating | last season's net rating per 100 possessions; its coefficient on this season's margin is fitted by OLS on the training seasons |
| Static SOS | `retro_*` uses opponents' final records (descriptive only); `causal_*` uses opponents' records before each game |
