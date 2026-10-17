"""Four-state Gaussian HMM market-regime detector.

Two inference modes matter for backtesting:

* **smoothed** (``hmmlearn.predict_proba``, forward–backward): the probability for day *t*
  uses observations *after* t. Fine for describing history, but look-ahead in a backtest.
* **filtered** (forward pass only): the probability for day *t* uses observations up to *t*.
  This is what a trader could have known in real time.

``tfg`` mode fits on the full sample and uses smoothed probabilities (as the thesis code
did); ``corrected`` mode fits on the training window only and uses filtered probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from regime_rl.config import HMMConfig

log = logging.getLogger(__name__)

FEATURES = ["ret", "vol_20", "ret_63", "ret_252"]
# States are named by ascending mean daily return.
LABELS_BY_RETURN = ("bear", "sideways", "volatile", "bull")


def regime_features(prices: pd.Series) -> pd.DataFrame:
    """The four daily observations the HMM is fitted on, from benchmark prices."""
    log_ret = np.log(prices).diff()
    return pd.DataFrame(
        {
            "ret": log_ret,
            "vol_20": log_ret.rolling(20).std(),
            "ret_63": prices.pct_change(63, fill_method=None),
            "ret_252": prices.pct_change(252, fill_method=None),
        }
    ).dropna()


@dataclass
class RegimeDetector:
    config: HMMConfig
    model: GaussianHMM | None = None
    labels: dict[int, str] | None = None  # hidden-state index -> label
    fit_end: pd.Timestamp | None = None
    log_likelihood: float | None = None
    occupancy: np.ndarray | None = field(default=None, repr=False)

    # -------------------------------------------------------------- fitting
    def fit(
        self,
        features: pd.DataFrame,
        fit_end: str | pd.Timestamp | None = None,
        n_restarts: int | None = None,
    ) -> RegimeDetector:
        """Fit on rows up to and including ``fit_end`` (all rows if ``None``).

        ``n_restarts`` overrides ``config.n_restarts`` (the best log-likelihood wins).
        """
        if self.config.n_states != len(LABELS_BY_RETURN):
            raise ValueError("Regime labelling assumes 4 states")
        train = features if fit_end is None else features.loc[: pd.Timestamp(fit_end)]
        if len(train) < 252:
            raise ValueError(f"Need at least 252 observations to fit the HMM, got {len(train)}")
        x = train[FEATURES].to_numpy()
        best, best_ll = None, -np.inf
        # EM only finds a local optimum; restart from several seeds and keep the best fit.
        for i in range(max(1, n_restarts or self.config.n_restarts)):
            model = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=self.config.random_state + i,
            )
            # hmmlearn logs a WARNING whenever one EM step lowers the likelihood by a rounding
            # error; across restarts that is pure noise, so it is silenced during the fit.
            hmm_log = logging.getLogger("hmmlearn.base")
            previous = hmm_log.level
            hmm_log.setLevel(logging.ERROR)
            try:
                model.fit(x)
            finally:
                hmm_log.setLevel(previous)
            ll = model.score(x)
            if ll > best_ll:
                best, best_ll = model, ll
        model = best
        self.log_likelihood = float(best_ll)
        if not model.monitor_.converged:
            log.warning(
                "HMM EM did not report convergence after %d iterations", model.monitor_.iter
            )
        occupancy = np.bincount(model.predict(x), minlength=model.n_components) / len(x)
        self.occupancy = occupancy
        if occupancy.min() < 0.02:
            log.warning(
                "HMM state with %.1f%% occupancy: regimes may be degenerate", 100 * occupancy.min()
            )
        order = np.argsort(model.means_[:, 0])
        self.model = model
        self.labels = {int(state): LABELS_BY_RETURN[rank] for rank, state in enumerate(order)}
        self.fit_end = train.index[-1]
        return self

    def _require_fit(self) -> GaussianHMM:
        if self.model is None or self.labels is None:
            raise RuntimeError("RegimeDetector is not fitted")
        return self.model

    # -------------------------------------------------------------- inference
    def _log_emissions(self, x: np.ndarray) -> np.ndarray:
        m = self._require_fit()
        return np.column_stack(
            [
                multivariate_normal(mean=m.means_[k], cov=m.covars_[k], allow_singular=True).logpdf(
                    x
                )
                for k in range(m.n_components)
            ]
        )

    def filtered(self, features: pd.DataFrame) -> pd.DataFrame:
        """P(state_t | obs_1..t) via the forward algorithm (no look-ahead)."""
        m = self._require_fit()
        x = features[FEATURES].to_numpy()
        log_b = self._log_emissions(x).reshape(len(x), -1)
        log_a = np.log(np.maximum(m.transmat_, 1e-300))
        alpha = np.empty_like(log_b)
        prev = np.log(np.maximum(m.startprob_, 1e-300)) + log_b[0]
        alpha[0] = prev - logsumexp(prev)
        for t in range(1, len(x)):
            prev = logsumexp(alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
            alpha[t] = prev - logsumexp(prev)
        return self._frame(np.exp(alpha), features.index)

    def smoothed(self, features: pd.DataFrame) -> pd.DataFrame:
        """P(state_t | obs_1..T) via forward–backward (uses future observations)."""
        m = self._require_fit()
        return self._frame(m.predict_proba(features[FEATURES].to_numpy()), features.index)

    def _frame(self, probs: np.ndarray, index: pd.Index) -> pd.DataFrame:
        cols = [f"p_{self.labels[k]}" for k in range(probs.shape[1])]
        df = pd.DataFrame(probs, index=index, columns=cols)
        return df[[f"p_{lab}" for lab in ("bull", "bear", "sideways", "volatile")]]

    # -------------------------------------------------------------- reporting
    def profile(self) -> dict[str, pd.DataFrame]:
        """Fitted means, standard deviations, transition matrix and expected durations."""
        m = self._require_fit()
        names = [self.labels[k] for k in range(m.n_components)]
        means = pd.DataFrame(m.means_, index=names, columns=FEATURES)
        stds = pd.DataFrame(
            np.sqrt(np.array([np.diag(c) for c in m.covars_])), index=names, columns=FEATURES
        )
        trans = pd.DataFrame(m.transmat_, index=names, columns=names)
        stay = np.diag(m.transmat_)
        duration = pd.Series(1.0 / np.maximum(1.0 - stay, 1e-12), index=names, name="expected_days")
        order = ["bull", "volatile", "sideways", "bear"]
        return {
            "means": means.loc[order],
            "stds": stds.loc[order],
            "transition": trans.loc[order, order],
            "duration": duration.loc[order],
        }


def monthly_regimes(daily_probs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-end probabilities: ``(unlagged, lagged_one_month)``."""
    raw = daily_probs.resample("ME").last()
    return raw, raw.shift(1).dropna()


def detect_regimes(
    benchmark_prices: pd.Series, cfg_hmm: HMMConfig, tfg: bool, train_end: str | pd.Timestamp
) -> tuple[RegimeDetector, pd.DataFrame]:
    """Fit the detector and return daily probabilities according to the fidelity mode."""
    feats = regime_features(benchmark_prices)
    # tfg: a single EM run from random_state, as in the thesis.
    det = RegimeDetector(cfg_hmm).fit(
        feats, fit_end=None if tfg else train_end, n_restarts=1 if tfg else None
    )
    probs = det.smoothed(feats) if tfg else det.filtered(feats)
    return det, probs
