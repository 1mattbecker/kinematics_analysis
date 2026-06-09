"""
encoding_methods.py — Generic per-unit encoding analysis.

Two first-order methods (OLS regression, Spearman correlation) that accept
any predictor column from all_counts_df and return a standard result table
compatible with PerUnitStatsRegistry.

Design principles
-----------------
- Trial filtering is the caller's responsibility, not the method's.
  Each AnalysisSpec carries a `trial_query` string that is applied by
  fit_encoding() BEFORE the method sees the data.
- All methods return the same standard schema so downstream consumers
  (registry, spatial encoder) are method-agnostic.
- log/zscore transforms are part of the spec, not hardcoded to RT.
- AnalysisResult stores only spec + unit-level stats (no trial_df).
  To recover filtered trials: all_counts_df.query(result.spec.trial_query)

Standard result schema (result.stats)
--------------------------------------
    session_prefix  str
    unit            str
    n_trials        int
    T               float  — t-statistic (OLS) or rho-derived t (Spearman)
    p               float  — two-sided p-value
    q               float  — BH-FDR adjusted p-value across all units
    coef            float  — OLS slope or Spearman rho
    sig_fdr         bool   — q < fdr_alpha

Usage
-----
    from encoding_methods import AnalysisSpec, AnalysisResult, fit_encoding

    spec = AnalysisSpec(
        name="ols_rt_response",
        predictor_col="reaction_time_firstmove",
        response_col="spike_count",
        method="ols",
        trial_query="reaction_time_firstmove.between(0.05, 1.0)",
        log_x=True,
        zscore_x=True,
    )

    result = fit_encoding(all_counts_df, spec)
    # result.stats  -> per-unit stats DataFrame
    # result.spec   -> the AnalysisSpec used to produce it

    # Register in the shared registry:
    reg.register_regression(
        spec.name, result.stats,
        t_col="T", p_col="p", q_col="q", coef_col="coef",
    )
    # Or use the convenience method:
    reg.register(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

try:
    from aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys import (
        get_session_prefix as _get_session_prefix_pkg,
    )
except ImportError:
    import re as _re

    def _get_session_prefix_pkg(s: str) -> str:
        return _re.sub(r"_\d{2}-\d{2}-\d{2}$", "", str(s))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _canon_unit(x) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def _add_session_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure session_prefix and canonical unit columns exist (returns a copy).

    Ephys tables carry raw ``session`` / ``unit_id`` columns that must be
    derived into the grouping columns. Tables from other modalities may already
    provide the grouping columns directly (e.g. ``session_prefix`` / ``roi``),
    in which case nothing is synthesized and the frame passes through unchanged.
    """
    out = df.copy()
    if "session_prefix" not in out.columns and "session" in out.columns:
        out["session_prefix"] = out["session"].astype(str).map(_get_session_prefix_pkg)
    if "unit" not in out.columns and "unit_id" in out.columns:
        out["unit"] = out["unit_id"].map(_canon_unit)
    return out


def _fdr_bh(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Return BH-FDR q-values; NaN where p is NaN."""
    q = np.full_like(pvals, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if valid.any():
        _, q[valid], _, _ = multipletests(pvals[valid], alpha=alpha, method="fdr_bh")
    return q


def _rho_to_t(rho: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Convert Spearman rho to approximate t-statistic (df = n-2)."""
    rho = np.asarray(rho, dtype=float)
    n = np.asarray(n, dtype=float)
    t = np.full_like(rho, np.nan)
    ok = np.isfinite(rho) & np.isfinite(n) & (n > 2) & (np.abs(rho) < 0.999999)
    denom = np.maximum(1.0 - rho[ok] ** 2, 1e-12)
    t[ok] = rho[ok] * np.sqrt((n[ok] - 2.0) / denom)
    return t


# ── AnalysisSpec ─────────────────────────────────────────────────────────────

@dataclass
class AnalysisSpec:
    """Full specification for a single per-unit encoding analysis.

    Carries both the statistical parameters and the trial-level filter so
    that results are fully reproducible from the spec alone.

    Parameters
    ----------
    name : str
        Registry key (e.g. "ols_rt_response"). Must be unique per registry.
    predictor_col : str
        Column in all_counts_df used as the x / independent variable.
    response_col : str
        Column in all_counts_df used as the y / dependent variable.
    method : "ols" | "spearman" | "glm"
        Fitting method.
    glm_family : str
        GLM family name, used only when method="glm". One of "poisson",
        "nb" (negative binomial), "gaussian". Default "poisson".
    trial_query : str
        pandas DataFrame.query() string applied to all_counts_df before fitting.
        Leave empty to use all rows.  Examples:
            "reaction_time_firstmove.between(0.05, 1.0)"
            "reaction_time_firstmove < 2.0 & first_move_peak_velocity > 0"
    log_x : bool
        Log-transform the predictor before fitting.  Recommended for RT.
    zscore_x : bool
        Z-score the (possibly log-transformed) predictor so that the OLS
        coefficient is in units of standard deviations.
    min_trials : int
        Minimum usable trials per unit; units with fewer are returned as NaN.
    fdr_alpha : float
        FDR threshold for sig_fdr flag and q-value computation.
    notes : str
        Free-text description stored for manifest / provenance.
    group_cols : tuple of str
        Columns that identify one statistical unit (the per-X loop). Defaults
        to ("session_prefix", "unit") for ephys. For another modality whose
        table names its unit column differently, pass e.g.
        ("session_prefix", "roi"). These names are carried through to the
        output stats schema.
    """

    name: str
    predictor_col: str
    response_col: str
    method: str = "ols"
    trial_query: str = ""
    log_x: bool = False
    zscore_x: bool = True
    min_trials: int = 20
    fdr_alpha: float = 0.05
    notes: str = ""
    glm_family: str = "poisson"
    group_cols: Tuple[str, ...] = ("session_prefix", "unit")

    def summary(self) -> str:
        parts = [
            f"name={self.name!r}",
            f"method={self.method}",
            f"x={self.predictor_col}",
            f"y={self.response_col}",
        ]
        if self.method == "glm":
            parts.append(f"family={self.glm_family}")
        if self.trial_query:
            parts.append(f"filter={self.trial_query!r}")
        if tuple(self.group_cols) != ("session_prefix", "unit"):
            parts.append(f"group_cols={tuple(self.group_cols)}")
        if self.log_x:
            parts.append("log_x=True")
        if not self.zscore_x:
            parts.append("zscore_x=False")
        parts.append(f"min_trials={self.min_trials}")
        if self.notes:
            parts.append(f"notes={self.notes!r}")
        return "AnalysisSpec(" + ", ".join(parts) + ")"


# ── AnalysisResult ────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Output of fit_encoding: spec + unit-level stats only.

    No trial_df stored — recover filtered trials on demand via:
        all_counts_df.query(result.spec.trial_query)

    Attributes
    ----------
    spec : AnalysisSpec
        The full specification used to produce this result.
    stats : pd.DataFrame
        Per-unit summary with columns: session_prefix, unit, n_trials,
        T, p, q, coef, sig_fdr.
    """

    spec: AnalysisSpec
    stats: pd.DataFrame

    def sig(self) -> pd.DataFrame:
        """Return only significant rows (sig_fdr == True)."""
        return self.stats.loc[self.stats["sig_fdr"]].copy()

    def n_sig(self) -> dict:
        s = self.stats
        pos = int(((s["T"] > 0) & s["sig_fdr"]).sum())
        neg = int(((s["T"] < 0) & s["sig_fdr"]).sum())
        return {"pos": pos, "neg": neg, "total": pos + neg}

    def __repr__(self) -> str:
        ns = self.n_sig()
        return (
            f"AnalysisResult({self.spec.name!r}: "
            f"{len(self.stats)} units, "
            f"sig +{ns['pos']}/-{ns['neg']})"
        )


# ── OLS ──────────────────────────────────────────────────────────────────────

def fit_ols(df: pd.DataFrame, spec: AnalysisSpec) -> AnalysisResult:
    """OLS regression: response_col ~ 1 + predictor_col, per (session_prefix, unit).

    df should already have trial filtering applied (via spec.trial_query or
    manually).  Adds session_prefix and unit columns if not present.

    Returns
    -------
    AnalysisResult with .stats in the standard schema (see module docstring).
    """
    counts = _add_session_prefix(df)
    rows = []

    for keys, g in counts.groupby(list(spec.group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.dropna(subset=[spec.predictor_col, spec.response_col])
        x_raw = g[spec.predictor_col].to_numpy(dtype=float)
        y = g[spec.response_col].to_numpy(dtype=float)

        if spec.log_x:
            valid = np.isfinite(x_raw) & (x_raw > 0) & np.isfinite(y)
        else:
            valid = np.isfinite(x_raw) & np.isfinite(y)

        x_raw, y = x_raw[valid], y[valid]
        n = int(x_raw.size)

        base = dict(zip(spec.group_cols, keys))
        base.update({"n_trials": n, "T": np.nan, "p": np.nan, "coef": np.nan})
        if n < spec.min_trials or np.nanstd(x_raw) == 0 or np.nanstd(y) == 0:
            rows.append(base)
            continue

        x = np.log(x_raw) if spec.log_x else x_raw.copy()
        if spec.zscore_x:
            mu, sd = np.nanmean(x), np.nanstd(x)
            if sd > 0:
                x = (x - mu) / sd

        try:
            res = sm.OLS(y, sm.add_constant(x)).fit()
            rows.append({**base,
                         "coef": float(res.params[1]),
                         "T":    float(res.tvalues[1]),
                         "p":    float(res.pvalues[1])})
        except Exception:
            rows.append(base)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["q"] = _fdr_bh(out["p"].to_numpy(), spec.fdr_alpha)
    out["sig_fdr"] = out["q"] < spec.fdr_alpha

    n_valid = int(out["T"].notna().sum())
    n_pos = int(((out["T"] > 0) & out["sig_fdr"]).sum())
    n_neg = int(((out["T"] < 0) & out["sig_fdr"]).sum())
    print(f"[{spec.name}] {len(out)} units, {n_valid} valid, "
          f"sig: +{n_pos} / -{n_neg} (FDR α={spec.fdr_alpha})")

    return AnalysisResult(spec=spec, stats=out)


# ── Spearman ─────────────────────────────────────────────────────────────────

def run_spearman(df: pd.DataFrame, spec: AnalysisSpec) -> AnalysisResult:
    """Spearman correlation: response_col ~ predictor_col, per (session_prefix, unit).

    Returns the same standard schema as fit_ols.
    T is derived from rho via the n-2 degrees-of-freedom approximation.
    coef = Spearman rho.

    Note: log_x and zscore_x are ignored for Spearman (rank-based).
    """
    counts = _add_session_prefix(df)
    rows = []

    for keys, g in counts.groupby(list(spec.group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.dropna(subset=[spec.predictor_col, spec.response_col])
        x = g[spec.predictor_col].to_numpy(dtype=float)
        y = g[spec.response_col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        n = int(x.size)

        base = dict(zip(spec.group_cols, keys))
        base.update({"n_trials": n, "T": np.nan, "p": np.nan, "coef": np.nan})
        if n < spec.min_trials or np.unique(x).size < 2 or np.unique(y).size < 2:
            rows.append(base)
            continue

        rho, p = spearmanr(x, y, nan_policy="omit")
        rows.append({**base, "coef": float(rho), "p": float(p)})

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["T"] = _rho_to_t(out["coef"].to_numpy(), out["n_trials"].to_numpy())
    out["q"] = _fdr_bh(out["p"].to_numpy(), spec.fdr_alpha)
    out["sig_fdr"] = out["q"] < spec.fdr_alpha

    n_valid = int(out["T"].notna().sum())
    n_pos = int(((out["T"] > 0) & out["sig_fdr"]).sum())
    n_neg = int(((out["T"] < 0) & out["sig_fdr"]).sum())
    print(f"[{spec.name}] {len(out)} units, {n_valid} valid, "
          f"sig: +{n_pos} / -{n_neg} (FDR α={spec.fdr_alpha})")

    return AnalysisResult(spec=spec, stats=out)


# ── GLM ───────────────────────────────────────────────────────────────────────

_GLM_FAMILIES = {
    "poisson":  lambda: sm.families.Poisson(),
    "nb":       lambda: sm.families.NegativeBinomial(),
    "gaussian": lambda: sm.families.Gaussian(),
}


def fit_glm(df: pd.DataFrame, spec: AnalysisSpec) -> AnalysisResult:
    """GLM regression: response_col ~ 1 + predictor_col, per (session_prefix, unit).

    Family is controlled by spec.glm_family ("poisson", "nb", "gaussian").
    For Poisson, response must be non-negative counts; the link is log so
    coef is in units of log(mean spike count) per SD of predictor.

    log_x and zscore_x apply to the predictor as in fit_ols.
    The response is NOT z-scored (GLM handles the mean-variance relationship).
    """
    if spec.glm_family not in _GLM_FAMILIES:
        raise ValueError(
            f"Unknown glm_family {spec.glm_family!r}. "
            f"Choose from: {list(_GLM_FAMILIES)}"
        )

    counts = _add_session_prefix(df)
    rows = []

    for keys, g in counts.groupby(list(spec.group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.dropna(subset=[spec.predictor_col, spec.response_col])
        x_raw = g[spec.predictor_col].to_numpy(dtype=float)
        y = g[spec.response_col].to_numpy(dtype=float)

        if spec.log_x:
            valid = np.isfinite(x_raw) & (x_raw > 0) & np.isfinite(y) & (y >= 0)
        else:
            valid = np.isfinite(x_raw) & np.isfinite(y) & (y >= 0)

        x_raw, y = x_raw[valid], y[valid]
        n = int(x_raw.size)

        base = dict(zip(spec.group_cols, keys))
        base.update({"n_trials": n, "T": np.nan, "p": np.nan, "coef": np.nan})
        if n < spec.min_trials or np.nanstd(x_raw) == 0:
            rows.append(base)
            continue

        x = np.log(x_raw) if spec.log_x else x_raw.copy()
        if spec.zscore_x:
            mu, sd = np.nanmean(x), np.nanstd(x)
            if sd > 0:
                x = (x - mu) / sd

        try:
            family = _GLM_FAMILIES[spec.glm_family]()
            res = sm.GLM(y, sm.add_constant(x), family=family).fit(disp=False)
            rows.append({**base,
                         "coef": float(res.params[1]),
                         "T":    float(res.tvalues[1]),
                         "p":    float(res.pvalues[1])})
        except Exception:
            rows.append(base)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["q"] = _fdr_bh(out["p"].to_numpy(), spec.fdr_alpha)
    out["sig_fdr"] = out["q"] < spec.fdr_alpha

    n_valid = int(out["T"].notna().sum())
    n_pos = int(((out["T"] > 0) & out["sig_fdr"]).sum())
    n_neg = int(((out["T"] < 0) & out["sig_fdr"]).sum())
    print(f"[{spec.name}] {len(out)} units, {n_valid} valid, "
          f"sig: +{n_pos} / -{n_neg} (FDR α={spec.fdr_alpha})")

    return AnalysisResult(spec=spec, stats=out)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def fit_encoding(all_counts_df: pd.DataFrame, spec: AnalysisSpec) -> AnalysisResult:
    """Apply trial filter from spec, then dispatch to OLS, Spearman, or GLM.

    Parameters
    ----------
    all_counts_df : full trial × unit table from ephys_utils.build_all_counts_df
    spec : AnalysisSpec describing what to fit and how to filter

    Returns
    -------
    AnalysisResult with .spec and .stats (per-unit summary).
    To recover filtered trials: all_counts_df.query(result.spec.trial_query)
    """
    if spec.trial_query:
        try:
            df = all_counts_df.query(spec.trial_query).copy()
        except Exception as e:
            raise ValueError(f"trial_query failed: {spec.trial_query!r}\n{e}") from e
    else:
        df = all_counts_df.copy()

    if spec.method == "ols":
        return fit_ols(df, spec)
    elif spec.method == "spearman":
        return run_spearman(df, spec)
    elif spec.method == "glm":
        return fit_glm(df, spec)
    else:
        raise ValueError(f"Unknown method {spec.method!r}. Use 'ols', 'spearman', or 'glm'.")
