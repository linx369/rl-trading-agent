"""regime_rl — regime-aware momentum strategy selection with a Gaussian HMM and PPO.

Final-year project (TFG), Universitat Pompeu Fabra, 2025/26 — Tom Owen Prats.
"""

from regime_rl.config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "1.0.0"

ACTIONS: dict[int, str] = {
    0: "Top 5 max-Sharpe",
    1: "Top 15 max-Sharpe",
    2: "Top 30 momentum",
    3: "Mega-cap momentum",
    4: "Sector-tilted",
    5: "Top 3 ultra-concentrated",
    6: "TLT defensive",
    7: "Mean-reversion mixed",
}
N_ACTIONS = len(ACTIONS)
