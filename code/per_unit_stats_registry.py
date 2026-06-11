"""
PerUnitStatsRegistry — a lightweight registry for per-unit t-statistics.

Normalizes per-unit effect-size tables from heterogeneous sources
(OLS regressions, Spearman, partial correlations, external encoding models)
into a common schema keyed by (session_prefix, unit), enabling
one-liner pairwise comparisons, screening, and spatial analyses.

Primary registration path: reg.register(AnalysisResult) from encoding_methods.
Legacy paths: register_regression (raw DataFrame) and register_sue (batch T_ columns).

Visualization is handled by encoding_plots (no matplotlib in this module):
    registry_heatmap        pairwise Spearman rho heatmap
    registry_upset          significance overlap UpSet plot
    registry_summary        t-stat histogram + Wilcoxon/binom tests
    registry_compare_plot   scatter + marginals for compare() output
    registry_plot_examples  trial-level scatter for auto-selected units
    example_unit_scatter    trial-level scatter for explicit (unit, session) list

Usage
-----
    from per_unit_stats_registry import PerUnitStatsRegistry
    from encoding_methods import AnalysisSpec, fit_encoding

    reg = PerUnitStatsRegistry(get_session_prefix=get_session_prefix)
    result = fit_encoding(df, AnalysisSpec(name="rt_ols", ...))
    reg.register(result)
    reg.register_sue(sue_plus)

    merged = reg.compare("rt_ols", "sue::T_baseline_hit_all")
    results = reg.screen("rt_ols", prefix="sue::")
"""

from __future__ import annotations

import re
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


# ============================================================
# Canonical column names used internally
# ============================================================
_KEY_COLS = ["session_prefix", "unit"]

# Every registered entry gets normalized to this schema.
_STAT_COLS = ["t", "p", "q", "coef", "sig_fdr", "n_trials"]


def _get_session_prefix_default(s: str) -> str:
    return re.sub(r'_\d{2}-\d{2}-\d{2}$', '', str(s))


def _canon_unit(x) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns q-values (NaN where p is NaN)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if m.any():
        _, q_vals, _, _ = multipletests(p[m], alpha=alpha, method='fdr_bh')
        q[m] = q_vals
    return q


class PerUnitStatsRegistry:
    """
    Registry of per-unit t-statistic tables, all keyed by (session_prefix, unit).

    Parameters
    ----------
    get_session_prefix : callable, optional
        Function mapping full session string -> session_prefix.
        Defaults to stripping the HH-MM-SS suffix.
    alpha : float
        Significance threshold for FDR correction (default 0.05).
    """

    def __init__(
        self,
        get_session_prefix: Optional[Callable[[str], str]] = None,
        alpha: float = 0.05,
    ):
        self._entries: Dict[str, pd.DataFrame] = {}
        self._meta: Dict[str, dict] = {}
        self._gsp = get_session_prefix or _get_session_prefix_default
        self.alpha = alpha

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------
    @property
    def names(self) -> List[str]:
        """List of registered entry names."""
        return sorted(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        lines = [f"PerUnitStatsRegistry ({len(self)} entries):"]
        for name in self.names:
            df = self._entries[name]
            n = len(df)
            n_sig = int(df["sig_fdr"].sum()) if "sig_fdr" in df.columns else "?"
            src = self._meta.get(name, {}).get("source", "unknown")
            lines.append(f"  {name:40s}  n={n:>4d}  sig={n_sig:>4}  source={src}")
        return "\n".join(lines)

    # ----------------------------------------------------------
    # Get / list
    # ----------------------------------------------------------
    def get(self, name: str) -> pd.DataFrame:
        """Return the normalized table for a registered entry."""
        if name not in self._entries:
            raise KeyError(
                f"'{name}' not in registry. Available: {self.names}"
            )
        return self._entries[name].copy()

    def list_by_source(self, source: str) -> List[str]:
        """List entry names that came from a given source tag."""
        return [n for n, m in self._meta.items() if m.get("source") == source]

    # ----------------------------------------------------------
    # Registration: OLS regression
    # ----------------------------------------------------------
    def register_regression(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        t_col: str,
        p_col: str,
        coef_col: Optional[str] = None,
        q_col: Optional[str] = None,
        n_col: Optional[str] = None,
        session_prefix_col: str = "session_prefix",
        unit_col: str = "unit",
        overwrite: bool = False,
    ) -> None:
        """
        Register a per-unit table from an OLS/GLM regression.

        Parameters
        ----------
        name : str
            Registry key (e.g. "ols_rt_response").
        df : DataFrame
            Must contain t_col, p_col, and key columns.
        t_col, p_col : str
            Column names for t-statistic and p-value.
        coef_col, q_col, n_col : str, optional
            Column names for coefficient, q-value, and trial count.
        """
        self._check_overwrite(name, overwrite)

        out = df.copy()
        out["session_prefix"] = out[session_prefix_col].astype(str)
        out["unit"] = out[unit_col].map(_canon_unit)

        out["t"] = out[t_col].astype(float)
        out["p"] = out[p_col].astype(float)

        if q_col and q_col in out.columns:
            out["q"] = out[q_col].astype(float)
        else:
            out["q"] = _fdr_bh(out["p"].values, self.alpha)

        out["sig_fdr"] = out["q"] < self.alpha

        if coef_col and coef_col in out.columns:
            out["coef"] = out[coef_col].astype(float)
        else:
            out["coef"] = np.nan

        if n_col and n_col in out.columns:
            out["n_trials"] = out[n_col].astype(float)
        else:
            out["n_trials"] = np.nan

        out = out.drop_duplicates(subset=_KEY_COLS, keep="first")
        self._store(name, out, source="regression")

    def register(self, result, *, overwrite: bool = False) -> None:
        """Register an AnalysisResult directly (convenience wrapper).

        Equivalent to register_regression(result.spec.name, result.stats, ...).
        The stats DataFrame is expected to already have the standard schema
        (T, p, q, coef, sig_fdr) as produced by fit_encoding.
        """
        self.register_regression(
            result.spec.name,
            result.stats,
            t_col="T",
            p_col="p",
            q_col="q",
            coef_col="coef",
            n_col="n_trials",
            overwrite=overwrite,
        )

    # ----------------------------------------------------------
    # Registration: Sue's encoding table (batch)
    # ----------------------------------------------------------
    def register_sue(
        self,
        sue_df: pd.DataFrame,
        *,
        t_prefix: str = "T_",
        p_prefix: str = "p_",
        coef_prefix: str = "coef_",
        registry_prefix: str = "sue",
        session_col: str = "session",
        unit_col: str = "unit",
        session_prefix_col: Optional[str] = "session_prefix",
        overwrite: bool = False,
    ) -> int:
        """
        Bulk-register all T_ columns from Sue's encoding table.

        Each T_{suffix} column becomes a registry entry named
        "{registry_prefix}::{suffix}".

        Returns
        -------
        int
            Number of entries registered.
        """
        df = sue_df.copy()

        if session_prefix_col and session_prefix_col in df.columns:
            df["session_prefix"] = df[session_prefix_col].astype(str)
        else:
            df["session_prefix"] = df[session_col].astype(str).map(self._gsp)
        df["unit"] = df[unit_col].map(_canon_unit)

        t_cols = [c for c in df.columns
                  if isinstance(c, str) and c.startswith(t_prefix)]

        count = 0
        for t_col in t_cols:
            suffix = t_col[len(t_prefix):]
            p_col = f"{p_prefix}{suffix}"
            coef_col = f"{coef_prefix}{suffix}"
            entry_name = f"{registry_prefix}::{suffix}"

            if not overwrite and entry_name in self._entries:
                continue

            out = df[_KEY_COLS].copy()
            out["t"]    = df[t_col].astype(float) if t_col in df.columns else np.nan
            out["p"]    = df[p_col].astype(float) if p_col in df.columns else np.nan
            out["coef"] = df[coef_col].astype(float) if coef_col in df.columns else np.nan
            out["q"] = _fdr_bh(out["p"].values, self.alpha)
            out["sig_fdr"] = out["q"] < self.alpha
            out["n_trials"] = np.nan

            out = out.drop_duplicates(subset=_KEY_COLS, keep="first")
            self._store(entry_name, out, source="sue")
            count += 1

        return count

    # ----------------------------------------------------------
    # Comparison: pairwise overlap stats (no plotting)
    # ----------------------------------------------------------
    def compare(
        self,
        name_a: str,
        name_b: str,
        *,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """Pairwise comparison of two registered entries.

        Returns a merged DataFrame with columns:
            session_prefix, unit, t_a, t_b, sig_fdr_a, sig_fdr_b, sig_category

        For visualization pass the result to encoding_plots.registry_compare_plot().

        Parameters
        ----------
        name_a, name_b : str
            Registry keys to compare.
        alpha : float, optional
            Override registry-level alpha for significance.
        """
        alpha = alpha or self.alpha
        a = self.get(name_a)
        b = self.get(name_b)

        merged = a[_KEY_COLS + ["t", "sig_fdr"]].merge(
            b[_KEY_COLS + ["t", "sig_fdr"]],
            on=_KEY_COLS,
            how="inner",
            suffixes=("_a", "_b"),
        )

        sa = merged["sig_fdr_a"]
        sb = merged["sig_fdr_b"]
        cats = pd.Series("neither", index=merged.index)
        cats[sa & sb]   = "both"
        cats[sa & ~sb]  = f"{name_a} only"
        cats[~sa & sb]  = f"{name_b} only"
        merged["sig_category"] = cats

        return merged.dropna(subset=["t_a", "t_b"]).reset_index(drop=True)

    # ----------------------------------------------------------
    # Screening: one entry vs many
    # ----------------------------------------------------------
    def screen(
        self,
        name: str,
        *,
        against: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        source: Optional[str] = None,
        min_n: int = 10,
        rank_by: str = "abs_rho",
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Screen one entry's t-stats against many others via Spearman
        correlation of t-values across units.

        Parameters
        ----------
        name : str
            The entry to screen.
        against : list of str, optional
            Explicit list of entry names to screen against.
        prefix : str, optional
            Screen against all entries whose name starts with this.
        source : str, optional
            Screen against all entries from this source (e.g. "sue").
        min_n : int
            Minimum number of overlapping units to compute correlation.
        rank_by : str
            Sort results by "abs_rho", "rho", or "p".
        top_n : int, optional
            Return only the top N results.

        Returns
        -------
        DataFrame with columns: entry, n, rho, p, abs_rho,
        fisher_OR, fisher_p, n_both, n_ref_only, n_tgt_only
        """
        from scipy.stats import fisher_exact

        targets = []
        if against:
            targets.extend(against)
        if prefix:
            targets.extend([n for n in self.names if n.startswith(prefix)])
        if source:
            targets.extend(self.list_by_source(source))
        if not targets:
            targets = [n for n in self.names if n != name]

        targets = sorted(set(t for t in targets if t != name))

        ref = self.get(name)
        rows = []

        for tgt in targets:
            tgt_df = self.get(tgt)
            merged = ref[_KEY_COLS + ["t", "sig_fdr"]].merge(
                tgt_df[_KEY_COLS + ["t", "sig_fdr"]],
                on=_KEY_COLS,
                how="inner",
                suffixes=("_ref", "_tgt"),
            ).dropna(subset=["t_ref", "t_tgt"])

            n = len(merged)
            if n < min_n:
                rows.append({"entry": tgt, "n": n, "rho": np.nan,
                             "p": np.nan, "abs_rho": np.nan,
                             "fisher_OR": np.nan, "fisher_p": np.nan,
                             "n_both": 0, "n_ref_only": 0, "n_tgt_only": 0})
                continue

            rho, p = spearmanr(merged["t_ref"], merged["t_tgt"])

            sa = merged["sig_fdr_ref"].values.astype(bool)
            sb = merged["sig_fdr_tgt"].values.astype(bool)
            contingency = np.array([
                [int((~sa & ~sb).sum()), int((~sa &  sb).sum())],
                [int(( sa & ~sb).sum()), int(( sa &  sb).sum())],
            ])
            odds_ratio, fisher_p = fisher_exact(contingency)

            rows.append({
                "entry": tgt,
                "n": n,
                "rho": float(rho),
                "p": float(p),
                "abs_rho": float(abs(rho)),
                "fisher_OR": float(odds_ratio),
                "fisher_p": float(fisher_p),
                "n_both": int((sa & sb).sum()),
                "n_ref_only": int((sa & ~sb).sum()),
                "n_tgt_only": int((~sa & sb).sum()),
            })

        result = pd.DataFrame(rows)
        if rank_by in result.columns:
            result = result.sort_values(rank_by, ascending=(rank_by == "p"))

        if top_n:
            result = result.head(top_n)

        return result.reset_index(drop=True)

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    def _check_overwrite(self, name: str, overwrite: bool) -> None:
        if name in self._entries and not overwrite:
            raise ValueError(
                f"'{name}' already registered. Pass overwrite=True to replace."
            )

    def _store(self, name: str, df: pd.DataFrame, source: str) -> None:
        keep = _KEY_COLS + [c for c in _STAT_COLS if c in df.columns]
        self._entries[name] = df[keep].reset_index(drop=True)
        self._meta[name] = {"source": source}
