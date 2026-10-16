"""Typed run configuration, loaded from YAML.

The single most important switch is ``fidelity``:

* ``"tfg"``       – reproduces the logic of the original thesis script (``legacy/``),
                    including its known defects, so the published numbers can be re-derived.
* ``"corrected"`` – the same strategy with those defects fixed (see ``docs/corrections.md``).

Every behavioural difference between the two modes is read from :attr:`Config.tfg`, so a
``grep "cfg.tfg"`` lists them all.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from regime_rl.markets import Market, get_market

Fidelity = Literal["tfg", "corrected"]


@dataclass(frozen=True)
class SplitConfig:
    val_start: str = "2021-01-01"
    test_start: str = "2023-01-01"


@dataclass(frozen=True)
class UniverseConfig:
    top_n_liquidity: int = 200
    liquidity_window_months: int = 60
    min_history_days: int = 50
    crash_threshold: float = -0.10  # exclude stocks down more than this last month
    min_names_after_filter: int = 5


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 4
    n_iter: int = 300
    tol: float = 1e-5
    random_state: int = 42
    covariance_type: str = "full"
    n_restarts: int = 10  # EM restarts (corrected mode only; tfg uses a single run)


@dataclass(frozen=True)
class GateConfig:
    bear_blocks_concentrated: float = 0.40  # P(Bear) above -> block actions 0, 1, 5
    volatile_blocks_top: float = 0.50  # P(Volatile) above -> block actions 0, 5
    bull_min_for_top3: float = 0.65  # P(Bull) below -> block action 5
    bull_blocks_tlt: float = 0.60  # P(Bull) above AND
    bear_max_for_tlt_block: float = 0.20  # P(Bear) below -> block action 6


@dataclass(frozen=True)
class RewardConfig:
    w_ir: float = 0.4
    w_calmar: float = 0.4
    w_crash: float = 0.2
    crash_threshold: float = -0.05
    crash_scale: float = 5.0
    repeat_penalty: float = 0.05
    window: int = 6
    clip: float = 3.0


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1e-4
    n_steps: int = 512
    batch_size: int = 128
    n_epochs: int = 15
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    ent_coef: float = 0.06
    net_arch: tuple[int, ...] = (256, 256, 128)
    passes: int = 100  # total timesteps = passes * n_train_months


@dataclass(frozen=True)
class SelectionConfig:
    seeds: tuple[int, ...] = (42, 7, 13, 99, 2025, 1337, 888, 314, 777, 555)
    softmax_temperature: float = 0.5
    max_ensemble: int = 3


@dataclass(frozen=True)
class RobustnessConfig:
    bootstrap_reps: int = 2000
    bootstrap_block: int = 6
    mc_paths: int = 1000
    mc_horizon_months: int = 24
    random_state: int = 42


@dataclass(frozen=True)
class Config:
    market_code: str = "US"
    fidelity: Fidelity = "corrected"
    start: str = "2006-01-01"
    end: str = "2026-03-31"
    data_dir: str = "data"
    output_dir: str = "outputs"
    transaction_cost: float = 0.0020
    split: SplitConfig = field(default_factory=SplitConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    def __post_init__(self) -> None:
        if self.fidelity not in ("tfg", "corrected"):
            raise ValueError(f"fidelity must be 'tfg' or 'corrected', got {self.fidelity!r}")
        get_market(self.market_code)  # validate early
        if not (self.start < self.split.val_start < self.split.test_start < self.end):
            raise ValueError("Expected start < val_start < test_start < end")

    @property
    def tfg(self) -> bool:
        """True when reproducing the original thesis logic (defects included)."""
        return self.fidelity == "tfg"

    @property
    def market(self) -> Market:
        return get_market(self.market_code)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / f"{self.market_code.lower()}_{self.fidelity}"

    def replace(self, **changes: Any) -> Config:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS: dict[str, type] = {
    "split": SplitConfig,
    "universe": UniverseConfig,
    "hmm": HMMConfig,
    "gates": GateConfig,
    "reward": RewardConfig,
    "ppo": PPOConfig,
    "selection": SelectionConfig,
    "robustness": RobustnessConfig,
}


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    cooked = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**cooked)


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw)
    if "market" in raw:
        raw["market_code"] = raw.pop("market")
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTIONS:
            kwargs[key] = _build_section(_SECTIONS[key], value or {})
        else:
            kwargs[key] = value
    top = {f.name for f in dataclasses.fields(Config)}
    unknown = set(kwargs) - top
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("start", "end"):
        if key in kwargs:
            kwargs[key] = str(kwargs[key])
    if "split" in kwargs:
        s = kwargs["split"]
        kwargs["split"] = SplitConfig(str(s.val_start), str(s.test_start))
    return Config(**kwargs)


def load_config(path: str | Path, **overrides: Any) -> Config:
    """Load a YAML config and apply top-level overrides (e.g. ``fidelity="tfg"``)."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return config_from_dict(raw)
