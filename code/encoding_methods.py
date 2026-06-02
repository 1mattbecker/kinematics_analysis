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

Standard result schema
----------------------
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
    from encoding_methods import AnalysisSpec, fit_encoding

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
    # result["results"]   -> per-unit stats DataFrame
    # result["trial_df"]  -> trial-level rows that passed the filter

    # Register in the shared registry:
    reg.register_regression(
        spec.name, result["results"],
        t_col="T", p_col="p", q_col="q", coef_col="coef",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
    """Add session_prefix and canonical unit columns in-place (copy)."""
    out = df.copy()
    out["session_prefix"] = out["session"].astype(str).map(_get_session_prefix_pkg)
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
    method : "ols" | "spearman"
        Fitting method.
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

    def summary(self) -> str:
        parts = [
            f"name={self.name!r}",
            f"method={self.method}",
            f"x={self.predictor_col}",
            f"y={self.response_col}",
        ]
        if self.trial_query:
            parts.append(f"filter={self.trial_query!r}")
        if self.log_x:
            parts.append("log_x=True")
        if not self.zscore_x:
            parts.append("zscore_x=False")
        parts.append(f"min_trials={self.min_trials}")
        if self.notes:
            parts.append(f"notes={self.notes!r}")
        return "AnalysisSpec(" + ", ".join(parts) + ")"


# ── OLS ──────────────────────────────────────────────────────────────────────

def fit_ols(df: pd.DataFrame, spec: AnalysisSpec) -> dict:
    """OLS regression: response_col ~ 1 + predictor_col, per (session_prefix, unit).

    df should already have trial filtering applied (via spec.trial_query or
    manually).  Adds session_prefix and unit columns if not present.

    Returns
    -------
    dict with keys:
        "results"   : DataFrame, standard schema (see module docstring)
        "trial_df"  : subset of df rows used in at least one unit fit
    """
    counts = _add_session_prefix(df)
    rows = []
    keep_idx = []

    for (sp, u), g in counts.groupby(["session_prefix", "unit"]):
        g = g.dropna(subset=[spec.predictor_col, spec.response_col])
        x_raw = g[spec.predictor_col].to_numpy(dtype=float)
        y = g[spec.response_col].to_numpy(dtype=float)

        if spec.log_x:
            valid = np.isfinite(x_raw) & (x_raw > 0) & np.isfinite(y)
        else:
            valid = np.isfinite(x_raw) & np.isfinite(y)

        keep_idx.extend(g.index[valid].tolist())
        x_raw, y = x_raw[valid], y[valid]
        n = int(x_raw.size)

        base = {"session_prefix": sp, "unit": u, "n_trials": n,
                "T": np.nan, "p": np.nan, "coef": np.nan}
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

    trial_df = counts.loc[counts.index.isin(keep_idx)].copy() if keep_idx else counts.iloc[:0].copy()
    return {"results": out, "trial_df": trial_df}


# ── Spearman ─────────────────────────────────────────────────────────────────

def run_spearman(df: pd.DataFrame, spec: AnalysisSpec) -> dict:
    """Spearman correlation: response_col ~ predictor_col, per (session_prefix, unit).

    Returns the same standard schema as fit_ols.
    T is derived from rho via the n-2 degrees-of-freedom approximation.
    coef = Spearman rho.

    Note: log_x and zscore_x are ignored for Spearman (rank-based).
    """
    counts = _add_session_prefix(df)
    rows = []
    keep_idx = []

    for (sp, u), g in counts.groupby(["session_prefix", "unit"]):
        g = g.dropna(subset=[spec.predictor_col, spec.response_col])
        x = g[spec.predictor_col].to_numpy(dtype=float)
        y = g[spec.response_col].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        keep_idx.extend(g.index[valid].tolist())
        x, y = x[valid], y[valid]
        n = int(x.size)

        base = {"session_prefix": sp, "unit": u, "n_trials": n,
                "T": np.nan, "p": np.nan, "coef": np.nan}
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

    trial_df = counts.loc[counts.index.isin(keep_idx)].copy() if keep_idx else counts.iloc[:0].copy()
    return {"results": out, "trial_df": trial_df}


# ── Dispatcher ────────────────────────────────────────────────────────────────

def fit_encoding(all_counts_df: pd.DataFrame, spec: AnalysisSpec) -> dict:
    """Apply trial filter from spec, then dispatch to OLS or Spearman.

    Parameters
    ----------
    all_counts_df : full trial × unit table from ephys_utils.build_all_counts_df
    spec : AnalysisSpec describing what to fit and how to filter

    Returns
    -------
    dict with "results" (per-unit stats) and "trial_df" (filtered trials used)
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
    else:
        raise ValueError(f"Unknown method {spec.method!r}. Use 'ols' or 'spearman'.")
