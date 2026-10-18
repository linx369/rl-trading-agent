"""Multi-seed PPO training, validation-based selection and policy evaluation.

Selection protocol (no test data is touched until the very last step):

1. Train one agent per seed on the training block.
2. Score each on the validation block (annualised return minus benchmark's).
3. Keep seeds at or above the median; take the best ``max_ensemble`` of them.
4. Weight them with a softmax (temperature T) over their validation alphas.
5. Keep whichever of {ensemble, single best seed} scores higher on validation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from regime_rl.config import Config
from regime_rl.env.trading_env import EpisodeData, TradingEnv

log = logging.getLogger(__name__)


def make_torch_deterministic() -> None:
    """Same seed, same agent, on every machine.

    Multi-threaded matrix maths (e.g. Apple's Accelerate on macOS) can add floating-point
    sums in a different order from run to run, which PPO then amplifies into different
    policies. The networks here are tiny, so one thread costs almost nothing.
    """
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_agent(data: EpisodeData, cfg: Config, seed: int):
    """Train one PPO agent; ``passes`` full sweeps over the training months."""
    from stable_baselines3 import PPO

    make_torch_deterministic()

    p = cfg.ppo
    env = TradingEnv(data, cfg)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        policy_kwargs={"net_arch": list(p.net_arch)},
        verbose=0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=max(p.passes * len(data.dates), p.n_steps), progress_bar=False)
    return model


def rollout(agent, data: EpisodeData, cfg: Config) -> pd.DataFrame:
    """Run a trained agent deterministically through ``data``; one row per month."""
    env = TradingEnv(data, cfg)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(action))
        rows.append(info)
    return pd.DataFrame(rows).set_index("date")


def annualised_alpha(rets: Sequence[float], bench: Sequence[float]) -> float:
    r, b = np.asarray(rets, float), np.asarray(bench, float)
    n = len(r)
    if n == 0:
        return float("nan")
    if n < 3:
        return float((r - b).mean() * 12)
    return float(np.prod(1 + r) ** (12 / n) - np.prod(1 + b) ** (12 / n))


def softmax_weights(alphas: Sequence[float], temperature: float) -> list[float]:
    a = np.asarray(alphas, float)
    e = np.exp((a - a.min()) / temperature)
    return list(e / e.sum())


@dataclass
class Policy:
    """A single agent or a return-weighted blend of agents."""

    agents: list
    seeds: list[int]
    weights: list[float]

    @property
    def is_ensemble(self) -> bool:
        return len(self.agents) > 1

    def run(self, data: EpisodeData, cfg: Config) -> pd.DataFrame:
        """Blended monthly record. Each agent runs its own portfolio; returns are weighted.

        The displayed ``action`` is the weight-majority vote across agents.
        """
        runs = [rollout(a, data, cfg) for a in self.agents]
        base = runs[0].copy()
        base["ret"] = sum(w * r["ret"] for w, r in zip(self.weights, runs, strict=True))
        base["excess"] = base["ret"] - base["bench_ret"]
        if self.is_ensemble:
            votes = pd.DataFrame({i: r["action"] for i, r in enumerate(runs)})
            base["action"] = votes.apply(
                lambda row: max(
                    set(row),
                    key=lambda a: sum(w for w, x in zip(self.weights, row, strict=True) if x == a),
                ),
                axis=1,
            )
        return base


@dataclass
class SelectionResult:
    policy: Policy
    seed_scores: pd.DataFrame  # seed, val_alpha, qualified, weight
    ensemble_val_alpha: float
    single_val_alpha: float
    agents: dict[int, object] = field(repr=False, default_factory=dict)


def select_policy(
    train: EpisodeData, val: EpisodeData, cfg: Config, model_dir: Path | None = None
) -> SelectionResult:
    sc = cfg.selection
    agents, scores = {}, []
    for seed in sc.seeds:
        log.info("Training seed %d (%d timesteps)…", seed, cfg.ppo.passes * len(train.dates))
        agent = train_agent(train, cfg, seed)
        rec = rollout(agent, val, cfg)
        a = annualised_alpha(rec["ret"], rec["bench_ret"])
        log.info("  seed %d: validation alpha %+.2f%%/yr", seed, 100 * a)
        agents[seed] = agent
        scores.append((seed, a))
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            agent.save(model_dir / f"seed_{seed}.zip")

    ranked = sorted(scores, key=lambda x: -x[1])
    median = float(np.median([a for _, a in ranked]))
    qualified = [(s, a) for s, a in ranked if a >= median][: sc.max_ensemble]
    if len(qualified) < 2:
        qualified = ranked[:1]
    weights = softmax_weights([a for _, a in qualified], sc.softmax_temperature)

    ens = Policy([agents[s] for s, _ in qualified], [s for s, _ in qualified], weights)
    single = Policy([agents[qualified[0][0]]], [qualified[0][0]], [1.0])
    ens_rec, single_rec = ens.run(val, cfg), single.run(val, cfg)
    ens_a = annualised_alpha(ens_rec["ret"], ens_rec["bench_ret"])
    single_a = annualised_alpha(single_rec["ret"], single_rec["bench_ret"])
    chosen = single if (single_a >= ens_a or not ens.is_ensemble) else ens
    log.info(
        "Validation: ensemble %+.2f%% vs single best %+.2f%% -> %s",
        100 * ens_a,
        100 * single_a,
        "ensemble" if chosen is ens else "single",
    )

    qual_w = dict(zip([s for s, _ in qualified], weights, strict=True))
    table = pd.DataFrame(ranked, columns=["seed", "val_alpha"])
    table["qualified"] = table["seed"].isin(qual_w)
    table["weight"] = table["seed"].map(qual_w).fillna(0.0)
    table["chosen"] = table["seed"].isin(chosen.seeds)
    return SelectionResult(
        policy=chosen,
        seed_scores=table,
        ensemble_val_alpha=ens_a,
        single_val_alpha=single_a,
        agents=agents,
    )
