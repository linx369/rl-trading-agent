# Legacy script

`rl_strategy_v11.py` is the original single-file research script behind the TFG results
(previously `rl_strategy-27.py` at the repository root). It is kept unchanged for reference
and is excluded from linting and tests.

Its logic is reproduced by the package with `fidelity: tfg`; see
[`docs/corrections.md`](../docs/corrections.md) for what the `corrected` mode changes and why.
Note that, as committed, it runs on the China universe (`MARKET = "CN"`) and trains a
single seed (`SEEDS = [2025]`).
