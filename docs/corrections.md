# Corrections: `tfg` vs `corrected`

The original thesis script ([`legacy/rl_strategy_v11.py`](../legacy/rl_strategy_v11.py)) was
refactored into the `regime_rl` package without changing its behaviour; a code review then
found the defects below. Rather than silently fixing them (which would make the thesis
numbers irreproducible), every fix sits behind one switch:

```yaml
fidelity: tfg        # reproduce the thesis logic, defects included
fidelity: corrected  # same strategy, defects fixed (default)
```

Every behavioural branch reads the `tfg` flag, so `grep -rn "tfg" src/` lists them all.
Each row names the test that pins the behaviour of both modes.

| # | Area | Original (`tfg`) | `corrected` | Likely effect on the backtest | Test |
|---|------|------------------|-------------|-------------------------------|------|
| 1 | **HMM look-ahead** | HMM fitted on the full 2006–2026 sample; probabilities from `predict_proba` (forward–backward *smoothing*), so the regime on day *t* uses data after *t*. The one-month lag does not remove this. | Fitted on the training window only (before `val_start`); probabilities from the forward pass (*filtering*), which uses data up to *t* only. | Optimistic: gates and regime features "knew" turning points early. | `test_hmm.py::test_filtered_probabilities_do_not_look_ahead`, `::test_smoothed_probabilities_do_look_ahead` |
| 2 | **Momentum window** | `ranked_momentum` applied daily lengths (252, 22, 63) to **monthly** prices: "momentum" = return from the start of the data to 21 months ago; crash filter on 21 months; relative-strength filter on 62 months. | 12-1 momentum on month-end prices (P<sub>t−1</sub>/P<sub>t−12</sub> − 1); crash filter on the last month; RS filter on 3 months. | Changes every momentum portfolio. | `test_selection.py::test_corrected_momentum_is_12_minus_1` |
| 3 | **Late listings dropped** | `dropna(axis=1)` over the whole history removed any ticker with a missing price since the data start (e.g. META, TSLA, ABBV). | Missing values only checked inside the window actually used. | Narrower, older universe. | `::test_tfg_momentum_drops_recent_listings`, `::test_corrected_momentum_includes_recent_listings` |
| 4 | **Liquidity filter** | `groupby(level=1).rank()` ranked dollar volume **within each ticker across time**, so "top 200" kept each stock's busiest months rather than the 200 most liquid stocks. | Ranked across tickers on each date; keep the top *N*. | Universe was not liquidity-filtered. | `test_panel.py::test_tfg_liquidity_ranks_each_stock_over_time` |
| 5 | **Sector tilt (action 4)** | Leading-sector names were duplicated in the list, but the weighting step de-duplicated them, so action 4 was identical to action 2. Sector chosen on last month's return. | Momentum weights × 2 for the leading sector, renormalised; sector chosen on 3-month return (as the thesis states). | Action 4 becomes a distinct strategy. | `test_selection.py::test_tfg_sector_tilt_is_identical_to_top30`, `::test_corrected_sector_tilt_doubles_leading_sector` |
| 6 | **ATR / MACD scaling** | Z-scored with each stock's **whole-sample** mean and std (look-ahead). | Trailing 252-day z-score. | Small look-ahead in two state features. | `test_indicators.py::test_full_zscore_does_look_ahead` |
| 7 | **"Beta"** | Rolling 12-month **correlation** with the benchmark. | Rolling 12-month beta (cov / var). | Feature meaning only. | `test_panel.py::test_tfg_beta_is_correlation_and_lagged` |
| 8 | **Reward regime signals** | Bonus thresholds such as "P(Bull) > 0.75" were applied to `(z + 3) / 6` of the *z-scored* probability, i.e. "1.5 std above its 3-year average", not a probability. | Bonuses use the actual (lagged) probabilities. | Different reward shaping. | `pipeline.build_dataset` |
| 9 | **Blocked actions** | Replaced by the allowed action with the nearest *index* (e.g. Top 3 → Sector-tilted), which is arbitrary because indices are not ordered by risk. | Replaced by Top 30 momentum, which is never gated. | Different fallback. | `test_gates_reward.py::test_resolve_action`, `test_env.py::test_blocked_action_is_replaced` |
| 10 | **Repeat penalty** | `prev_actions` was never appended to, so the penalty never fired. | Penalises the 4th consecutive identical action. | Slightly more diversity in choices. | `test_env.py::test_repeat_penalty_only_in_corrected_mode` |
| 11 | **Episode length** | Episodes ended one month early (the last month was never traded or scored). | All months are used. | One extra test month. | `test_env.py::test_episode_length` |
| 12 | **Monte Carlo benchmark** | Two-year benchmark target = mean monthly return × 24 (not compounded). | Compounded. | P(beat benchmark) slightly lower. | `test_evaluation.py::test_monte_carlo` |
| 13 | **EM restarts** | One EM run from `random_state=42`. EM only finds a local optimum; on the synthetic test market a single run left one state almost empty. | Best of `hmm.n_restarts` (default 10) runs by log-likelihood; a warning is logged if any state has < 2% occupancy. | More stable regime definitions. | `test_hmm.py::test_restarts_never_worse_than_single_fit` |

## Not fidelity-dependent

These were fixed outright because they affect reproducibility, not methodology:

* **Data window** is fixed in the config (`start`, `end`) instead of "today minus 20 years".
* **Market** is chosen per config file; the original repo defaulted to `MARKET = "CN"`.
* **Seeds**: the original repo had `SEEDS = [2025]` (only one seed trained). The configs
  restore the ten seeds described in the thesis; use `--seeds 2025` to train just one.
* **No double training**: the original retrained the top-3 seeds a second time (identical
  results with the same seed); agents are now trained once and reused.
* **Ticker lists**: Siemens (SIE.DE) removed from the China industrials sector map; in the EU
  lists, CAPG.PA (duplicate of CAP.PA), BT-A.L and the delisted FCA.MI were removed and ABB
  is referenced by its Swiss listing ABBN.SW.

## Unchanged known limitations

* **Survivorship bias** (partly addressed, opt-in): by default the universe is today's S&P 500
  constituents. `universe.point_in_time: true` (see `configs/us_pit.yaml`) restricts each stock
  to its actual membership dates and adds former members that still trade. Delisted companies
  remain missing because Yahoo has no prices for them. It is independent of `fidelity`.
* **Flat 20 bp cost** per month regardless of turnover.
* **No risk-free rate** in the Sharpe ratio; "alpha" is the difference in annualised returns.
* **Gates** in the code differ from the thresholds written in Section 3.4 of the thesis (no
  dispersion condition; Top 3 needs P(Bull) ≥ 0.65). The code's thresholds are kept in both
  modes and are configurable under `gates:`.
