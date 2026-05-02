"""
PerUnitStatsRegistry — a lightweight registry for per-unit t-statistics.

Normalizes per-unit effect-size tables from heterogeneous sources
(Spearman correlations, OLS regressions, external encoding models)
into a common schema keyed by (session_prefix, unit), enabling
one-liner pairwise comparisons, screening, and spatial analyses.

Usage
-----
    from per_unit_stats_registry import PerUnitStatsRegistry

    reg = PerUnitStatsRegistry(get_session_prefix=get_session_prefix)

    # Register from your correlation pipeline
    reg.register_cor_df("rt_response", spkct_rt["cor_df"])
    reg.register_cor_df("rt_baseline", spkct_rt_bl["cor_df"])

    # Register from Sue's encoding table
    reg.register_sue(sue_plus)

    # Compare any two entries
    reg.compare("rt_response", "sue::T_baseline_hit_all")
    reg.screen("rt_response", prefix="sue::")
"""

from __future__ import annotations

import re
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


# ============================================================
# Canonical column names used internally
# ============================================================
_KEY_COLS = ["session_prefix", "unit"]

# Every registered entry gets normalized to this schema.
# Only t and p are required; coef and q are optional.
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
    Registry of per-unit t-statistic tables, all keyed by
    (session_prefix, unit).

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
    # Registration: cor_df (from run_and_plot_for_predictor)
    # ----------------------------------------------------------
    def register_cor_df(
        self,
        name: str,
        cor_df: Union[pd.DataFrame, dict],
        *,
        session_col: str = "session",
        unit_col: str = "unit_id",
        overwrite: bool = False,
    ) -> None:
        """
        Register a per-unit correlation table produced by
        `analyze_unit_correlations` / `run_and_plot_for_predictor`.

        Expects columns: spearman_rho, spearman_p, n_trials, and
        optionally spearman_q, spearman_t.

        Parameters
        ----------
        name : str
            Registry key (e.g. "rt_response", "velocity_partial").
        cor_df : DataFrame or dict
            If dict, expects key "cor_df" (the convention from
            run_and_plot_for_predictor return dicts).
        """
        if isinstance(cor_df, dict):
            cor_df = cor_df["cor_df"]

        self._check_overwrite(name, overwrite)

        df = cor_df.copy()

        # Build keys
        df["session_prefix"] = df[session_col].astype(str).map(self._gsp)
        df["unit"] = df[unit_col].map(_canon_unit)

        # Resolve t-stat: prefer pre-computed, else derive from rho + n
        if "spearman_t" in df.columns:
            df["t"] = df["spearman_t"].astype(float)
        elif "spearman_rho" in df.columns and "n_trials" in df.columns:
            r = df["spearman_rho"].to_numpy(dtype=float)
            n = df["n_trials"].to_numpy(dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                df["t"] = r * np.sqrt((n - 2) / (1 - r**2))
        else:
            raise ValueError(
                "cor_df must have 'spearman_t' or both "
                "'spearman_rho' and 'n_trials'"
            )

        # p-value
        if "spearman_p" in df.columns:
            df["p"] = df["spearman_p"].astype(float)
        else:
            df["p"] = np.nan

        # q-value (FDR)
        if "spearman_q" in df.columns:
            df["q"] = df["spearman_q"].astype(float)
        else:
            df["q"] = _fdr_bh(df["p"].values, self.alpha)

        # significance flag
        df["sig_fdr"] = df["q"] < self.alpha

        # optional columns
        if "n_trials" in df.columns:
            df["n_trials"] = df["n_trials"].astype(float)
        else:
            df["n_trials"] = np.nan

        df["coef"] = df.get("spearman_rho", pd.Series(np.nan, index=df.index)).astype(float)

        # Deduplicate on (session_prefix, unit) — keep first if multiple
        # sessions map to same prefix (shouldn't happen with 1 session/day)
        before = len(df)
        df = df.drop_duplicates(subset=_KEY_COLS, keep="first")
        if len(df) < before:
            warnings.warn(
                f"register_cor_df('{name}'): dropped {before - len(df)} "
                f"duplicate (session_prefix, unit) rows"
            )

        self._store(name, df, source="cor_df")

    # ----------------------------------------------------------
    # Registration: partial correlation
    # ----------------------------------------------------------
    def register_partial(
        self,
        name: str,
        partial_df: pd.DataFrame,
        *,
        session_col: str = "session",
        unit_col: str = "unit_id",
        overwrite: bool = False,
    ) -> None:
        """
        Register output of `analyze_unit_partial_correlations`.
        Same column expectations as cor_df.
        """
        # Partial correlation tables have the same schema as cor_df
        self.register_cor_df(
            name, partial_df,
            session_col=session_col,
            unit_col=unit_col,
            overwrite=overwrite,
        )
        self._meta[name]["source"] = "partial"

    # ----------------------------------------------------------
    # Registration: OLS regression (like build_sue_plus_with_rt_models)
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

        Parameters
        ----------
        sue_df : DataFrame
            Sue's per-unit features table.
        registry_prefix : str
            Prefix for registry names (default "sue").

        Returns
        -------
        int
            Number of entries registered.
        """
        df = sue_df.copy()

        # Build keys
        if session_prefix_col and session_prefix_col in df.columns:
            df["session_prefix"] = df[session_prefix_col].astype(str)
        else:
            df["session_prefix"] = df[session_col].astype(str).map(self._gsp)
        df["unit"] = df[unit_col].map(_canon_unit)

        # Find all T_ columns
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
            out["t"] = df[t_col].astype(float) if t_col in df.columns else np.nan
            out["p"] = df[p_col].astype(float) if p_col in df.columns else np.nan
            out["coef"] = df[coef_col].astype(float) if coef_col in df.columns else np.nan

            out["q"] = _fdr_bh(out["p"].values, self.alpha)
            out["sig_fdr"] = out["q"] < self.alpha
            out["n_trials"] = np.nan

            out = out.drop_duplicates(subset=_KEY_COLS, keep="first")
            self._store(entry_name, out, source="sue")
            count += 1

        return count

    # ----------------------------------------------------------
    # Comparison: pairwise scatter + overlap stats
    # ----------------------------------------------------------
    def compare(
        self,
        name_a: str,
        name_b: str,
        *,
        alpha: Optional[float] = None,
        show: bool = True,
        figsize: Tuple[float, float] = (7, 7),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pairwise comparison of two registered entries.
 
        Returns a merged DataFrame with columns:
            session_prefix, unit, t_a, t_b, sig_a, sig_b, sig_category
 
        If show=True, plots a scatter of t_a vs t_b colored by
        significance category, with marginal histograms.
 
        Parameters
        ----------
        name_a, name_b : str
            Registry keys to compare.
        alpha : float, optional
            Override registry-level alpha for significance.
        show : bool
            Whether to plot (default True).
        """
        import matplotlib.pyplot as plt
 
        alpha = alpha or self.alpha
        a = self.get(name_a)
        b = self.get(name_b)
 
        merged = a[_KEY_COLS + ["t", "sig_fdr"]].merge(
            b[_KEY_COLS + ["t", "sig_fdr"]],
            on=_KEY_COLS,
            how="inner",
            suffixes=("_a", "_b"),
        )
 
        # Significance categories
        sa = merged["sig_fdr_a"]
        sb = merged["sig_fdr_b"]
        cats = pd.Series("neither", index=merged.index)
        cats[sa & sb] = "both"
        cats[sa & ~sb] = f"{name_a} only"
        cats[~sa & sb] = f"{name_b} only"
        merged["sig_category"] = cats
 
        # Drop rows with NaN t-stats
        merged = merged.dropna(subset=["t_a", "t_b"]).reset_index(drop=True)
 
        # Population-level stats
        n = len(merged)
        rho, rho_p = spearmanr(merged["t_a"], merged["t_b"]) if n >= 3 else (np.nan, np.nan)
 
        counts = merged["sig_category"].value_counts()
 
        # Fisher exact test: sig overlap vs independence
        from scipy.stats import fisher_exact
        sa_arr = merged["sig_fdr_a"].values
        sb_arr = merged["sig_fdr_b"].values
        contingency = np.array([
            [int((~sa_arr & ~sb_arr).sum()), int((~sa_arr &  sb_arr).sum())],
            [int(( sa_arr & ~sb_arr).sum()), int(( sa_arr &  sb_arr).sum())],
        ])
        odds_ratio, fisher_p = fisher_exact(contingency)
 
        if show:
            fig, ax_main, ax_top, ax_right = self._scatter_with_marginals(
                merged["t_a"].values,
                merged["t_b"].values,
                merged["sig_category"].values,
                sig_x=merged["sig_fdr_a"].values,
                sig_y=merged["sig_fdr_b"].values,
                x_label=name_a,
                y_label=name_b,
                title=title or f"{name_a} vs {name_b}  (n={n})",
                rho=rho,
                rho_p=rho_p,
                fisher_or=odds_ratio,
                fisher_p=fisher_p,
                figsize=figsize,
            )
            if save_path is not None:
                self._save_fig(fig, save_path)
            plt.show()
 
        return merged

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
        DataFrame with columns: entry, n, rho, p, abs_rho
        """
        # Resolve target list
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
            merged = ref[_KEY_COLS + ["t"]].merge(
                tgt_df[_KEY_COLS + ["t"]],
                on=_KEY_COLS,
                how="inner",
                suffixes=("_ref", "_tgt"),
            ).dropna(subset=["t_ref", "t_tgt"])

            n = len(merged)
            if n < min_n:
                rows.append({"entry": tgt, "n": n, "rho": np.nan,
                             "p": np.nan, "abs_rho": np.nan})
                continue

            rho, p = spearmanr(merged["t_ref"], merged["t_tgt"])
            rows.append({
                "entry": tgt,
                "n": n,
                "rho": float(rho),
                "p": float(p),
                "abs_rho": float(abs(rho)),
            })

        result = pd.DataFrame(rows)
        if rank_by in result.columns:
            ascending = rank_by == "p"
            result = result.sort_values(rank_by, ascending=ascending)

        if top_n:
            result = result.head(top_n)

        return result.reset_index(drop=True)

    # ----------------------------------------------------------
    # Heatmap: pairwise Spearman correlation matrix of t-stats
    # ----------------------------------------------------------
    def heatmap(
        self,
        entries: Optional[List[str]] = None,
        *,
        source: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        cluster: bool = True,
        annot: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute and plot a pairwise Spearman correlation matrix of t-stats
        across all specified registry entries.
 
        Parameters
        ----------
        entries : list of str, optional
            Registry keys to include. If None, uses all entries
            (or filtered by `source`).
        source : str, optional
            If entries is None, include only entries from this source.
        labels : dict, optional
            Mapping from entry name to short display label.
            If None, uses entry names directly.
        cluster : bool
            Whether to reorder rows/cols by hierarchical clustering.
        annot : bool
            Whether to annotate cells with ρ values.
        figsize : tuple, optional
            Figure size. If None, auto-scaled from number of entries.
        title : str, optional
            Plot title.
        save_path : str, optional
            If provided, save .png and .svg (pass path without extension).
        show : bool
            Whether to display the figure.
        vmin, vmax : float
            Color scale limits (default -1 to 1).
 
        Returns
        -------
        DataFrame
            Symmetric correlation matrix (entry × entry).
        """
        import matplotlib.pyplot as plt
 
        # Resolve entry list
        if entries is None:
            if source is not None:
                entries = self.list_by_source(source)
            else:
                entries = self.names
        if len(entries) < 2:
            raise ValueError("Need at least 2 entries for a heatmap")
 
        n_entries = len(entries)
 
        # Build the t-stat matrix: units × entries (inner join across all)
        # Use pairwise correlations to handle different unit sets per entry
        rho_mat = np.full((n_entries, n_entries), np.nan)
        p_mat = np.full((n_entries, n_entries), np.nan)
        n_mat = np.full((n_entries, n_entries), 0, dtype=int)
 
        for i in range(n_entries):
            rho_mat[i, i] = 1.0
            p_mat[i, i] = 0.0
            df_i = self.get(entries[i])
            for j in range(i + 1, n_entries):
                df_j = self.get(entries[j])
                merged = df_i[_KEY_COLS + ["t"]].merge(
                    df_j[_KEY_COLS + ["t"]],
                    on=_KEY_COLS,
                    how="inner",
                    suffixes=("_i", "_j"),
                ).dropna(subset=["t_i", "t_j"])
 
                n_overlap = len(merged)
                n_mat[i, j] = n_mat[j, i] = n_overlap
 
                if n_overlap >= 3:
                    rho, p = spearmanr(merged["t_i"], merged["t_j"])
                    rho_mat[i, j] = rho_mat[j, i] = rho
                    p_mat[i, j] = p_mat[j, i] = p
 
        # Display labels
        if labels is None:
            labels = {e: e for e in entries}
        display_names = [labels.get(e, e) for e in entries]
 
        rho_df = pd.DataFrame(rho_mat, index=display_names, columns=display_names)
 
        # Optional clustering
        order = np.arange(n_entries)
        if cluster and n_entries >= 3:
            try:
                from scipy.cluster.hierarchy import linkage, leaves_list
                from scipy.spatial.distance import squareform
                dist = 1 - np.abs(rho_mat)
                np.fill_diagonal(dist, 0)
                # Handle NaN distances
                dist = np.nan_to_num(dist, nan=1.0)
                dist = (dist + dist.T) / 2
                condensed = squareform(dist, checks=False)
                Z = linkage(condensed, method="average")
                order = leaves_list(Z)
            except Exception:
                pass  # fall back to input order
 
        rho_ordered = rho_mat[np.ix_(order, order)]
        p_ordered = p_mat[np.ix_(order, order)]
        names_ordered = [display_names[i] for i in order]
 
        rho_df_ordered = pd.DataFrame(
            rho_ordered, index=names_ordered, columns=names_ordered
        )
 
        # Plot
        if figsize is None:
            side = max(6, 0.7 * n_entries + 2)
            figsize = (side, side)
 
        fig, ax = plt.subplots(figsize=figsize)
 
        im = ax.imshow(
            rho_ordered, cmap="RdBu_r", vmin=vmin, vmax=vmax,
            aspect="equal",
        )
 
        ax.set_xticks(range(n_entries))
        ax.set_yticks(range(n_entries))
        ax.set_xticklabels(names_ordered, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names_ordered, fontsize=8)
 
        # Annotate cells
        if annot:
            for i in range(n_entries):
                for j in range(n_entries):
                    if i == j:
                        continue
                    rho_val = rho_ordered[i, j]
                    p_val = p_ordered[i, j]
                    if not np.isfinite(rho_val):
                        continue
                    # Bold/star for significant
                    star = "*" if p_val < 0.05 else ""
                    color = "white" if abs(rho_val) > 0.6 else "black"
                    ax.text(
                        j, i, f"{rho_val:.2f}{star}",
                        ha="center", va="center",
                        fontsize=7, color=color,
                    )
 
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
        ax.set_title(title or "Pairwise t-stat correlations", fontsize=12)
 
        plt.tight_layout()
 
        if save_path is not None:
            self._save_fig(fig, save_path)
 
        if show:
            plt.show()
        else:
            plt.close(fig)
 
        return rho_df_ordered


    # ----------------------------------------------------------
    # UpSet plot: significance overlap across entries
    # ----------------------------------------------------------
    def upset(
        self,
        entries: List[str],
        *,
        labels: Optional[Dict[str, str]] = None,
        min_subset_size: int = 0,
        max_subsets: int = 25,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        sort_by: str = "cardinality",
    ) -> pd.DataFrame:
        """
        UpSet plot showing how many units are significant for each
        combination of registered entries.
 
        Parameters
        ----------
        entries : list of str
            Registry keys to include (2–8 recommended).
        labels : dict, optional
            Mapping from entry name to short display label.
        min_subset_size : int
            Hide intersections with fewer than this many units.
        max_subsets : int
            Maximum number of intersection bars to show.
        figsize : tuple, optional
            Figure size. If None, auto-scaled.
        title : str, optional
            Plot title.
        save_path : str, optional
            If provided, save .png and .svg (path without extension).
        show : bool
            Whether to display the figure.
        sort_by : str
            "cardinality" (largest first) or "degree" (by number of
            sets in the intersection).
 
        Returns
        -------
        DataFrame
            Membership table: one row per unit, boolean columns for
            each entry's significance, plus session_prefix and unit.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
 
        if len(entries) < 2:
            raise ValueError("Need at least 2 entries for an UpSet plot")
 
        # Build display labels
        if labels is None:
            labels = {e: e for e in entries}
        display_names = [labels.get(e, e) for e in entries]
 
        # Collect sig_fdr per entry, keyed by (session_prefix, unit)
        frames = []
        for entry_name, display in zip(entries, display_names):
            df = self.get(entry_name)
            sig_col = df[["session_prefix", "unit", "sig_fdr"]].copy()
            sig_col = sig_col.rename(columns={"sig_fdr": display})
            frames.append(sig_col)
 
        # Merge all on (session_prefix, unit) — outer join
        membership = frames[0]
        for f in frames[1:]:
            membership = membership.merge(f, on=_KEY_COLS, how="inner")
 
        for col in display_names:
            membership[col] = membership[col].fillna(False).astype(bool)
 
        n_total = len(membership)
        n_any_sig = int(membership[display_names].any(axis=1).sum())
 
        # Compute intersection counts
        # Each unique combination of True/False across display_names is a subset
        combo_cols = membership[display_names]
        combo_tuples = [tuple(row) for row in combo_cols.values]
        from collections import Counter
        combo_counts = Counter(combo_tuples)
 
        # Build subset table
        subsets = []
        for combo, count in combo_counts.items():
            if count < min_subset_size:
                continue
            degree = sum(combo)
            subsets.append({
                "combo": combo,
                "count": count,
                "degree": degree,
            })
 
        if sort_by == "cardinality":
            subsets.sort(key=lambda s: -s["count"])
        elif sort_by == "degree":
            subsets.sort(key=lambda s: (s["degree"], -s["count"]))
 
        subsets = subsets[:max_subsets]
        n_subsets = len(subsets)
        n_sets = len(display_names)
 
        # --- Plot ---
        if figsize is None:
            figsize = (max(8, n_subsets * 0.5 + 2), max(5, n_sets * 0.6 + 3))
 
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            2, 1, figure=fig,
            height_ratios=(3, n_sets * 0.6),
            hspace=0.05,
        )
 
        ax_bars = fig.add_subplot(gs[0])
        ax_dots = fig.add_subplot(gs[1], sharex=ax_bars)
 
        x = np.arange(n_subsets)
        counts = [s["count"] for s in subsets]
 
        # Bar chart
        ax_bars.bar(x, counts, color="steelblue", edgecolor="white", linewidth=0.5)
        for xi, c in zip(x, counts):
            ax_bars.text(xi, c + max(counts) * 0.02, str(c),
                         ha="center", va="bottom", fontsize=8)
        ax_bars.set_ylabel("units")
        ax_bars.set_xlim(-0.6, n_subsets - 0.4)
        ax_bars.set_ylim(0, max(counts) * 1.15)
        plt.setp(ax_bars.get_xticklabels(), visible=False)
        ax_bars.tick_params(axis="x", length=0)
 
        # Dot matrix
        dot_color_on = "#333333"
        dot_color_off = "#d0d0d0"
        line_color = "#333333"
 
        for i, s in enumerate(subsets):
            combo = s["combo"]
            active = [j for j, v in enumerate(combo) if v]
 
            # Draw all dots (off)
            for j in range(n_sets):
                ax_dots.scatter(
                    i, j, s=80, zorder=3,
                    color=dot_color_off, edgecolors="none",
                )
 
            # Draw active dots (on)
            for j in active:
                ax_dots.scatter(
                    i, j, s=80, zorder=4,
                    color=dot_color_on, edgecolors="none",
                )
 
            # Connect active dots with a line
            if len(active) > 1:
                ax_dots.plot(
                    [i, i], [min(active), max(active)],
                    color=line_color, linewidth=1.5, zorder=2,
                )
 
        ax_dots.set_yticks(range(n_sets))
        ax_dots.set_yticklabels(display_names, fontsize=9)
        ax_dots.set_ylim(-0.5, n_sets - 0.5)
        ax_dots.invert_yaxis()
        ax_dots.set_xticks([])
        ax_dots.set_xlim(-0.6, n_subsets - 0.4)
 
        # Clean up spines
        for spine in ax_dots.spines.values():
            spine.set_visible(False)
        ax_dots.tick_params(axis="both", length=0)
        ax_dots.set_facecolor("white")
 
        # Add set size annotation to the right of the dot matrix
        set_sizes = [int(membership[col].sum()) for col in display_names]
        for j, sz in enumerate(set_sizes):
            ax_dots.text(
                n_subsets - 0.3, j, f"n={sz}",
                ha="left", va="center", fontsize=8, color="grey",
            )
 
        if title:
            fig.suptitle(f"{title}  (n={n_total})", fontsize=12, y=0.98)
        else:
            fig.suptitle(
                f"Significance overlap (n={n_total}, any sig={n_any_sig})",
                fontsize=12, y=0.98,
            )
 
        if save_path is not None:
            self._save_fig(fig, save_path)
 
        if show:
            plt.show()
        else:
            plt.close(fig)
 
        return membership

    # ----------------------------------------------------------
    # Summary: t-stat distribution for one entry
    # ----------------------------------------------------------
    def summary(
        self,
        name: str,
        *,
        show: bool = True,
        bins: int = 30,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Summary statistics and histogram for one entry's t-stats.

        Tests:
            1. Wilcoxon signed-rank test: is the t-stat distribution
               centered at zero?
            2. Binomial test: among significant units, is the ratio of
               positive to negative different from 50/50?

        Returns dict with: n, n_sig, frac_sig, mean_t, median_t,
        n_pos_sig, n_neg_sig, wilcoxon_stat, wilcoxon_p,
        binom_p.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import wilcoxon, binomtest

        df = self.get(name)
        t = df["t"].dropna()
        sig = df.loc[df["t"].notna(), "sig_fdr"]

        n_pos_sig = int((sig & (t > 0)).sum())
        n_neg_sig = int((sig & (t < 0)).sum())
        n_sig = n_pos_sig + n_neg_sig

        # Wilcoxon signed-rank: t-stats centered at 0?
        t_arr = t.to_numpy()
        if len(t_arr) >= 10:
            w_stat, w_p = wilcoxon(t_arr)
        else:
            w_stat, w_p = np.nan, np.nan

        # Binomial: pos/neg ratio among sig units
        if n_sig >= 1:
            binom_result = binomtest(n_pos_sig, n_sig, 0.5)
            binom_p = binom_result.pvalue
        else:
            binom_p = np.nan

        stats = {
            "n": len(t),
            "n_sig": n_sig,
            "frac_sig": float(sig.mean()) if len(sig) > 0 else 0.0,
            "mean_t": float(t.mean()),
            "median_t": float(t.median()),
            "n_pos_sig": n_pos_sig,
            "n_neg_sig": n_neg_sig,
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
            "binom_p": float(binom_p) if np.isfinite(binom_p) else np.nan,
        }

        if show:
            fig, ax = plt.subplots(figsize=(6, 4))

            t_all = t.to_numpy()
            sig_arr = sig.to_numpy()

            bin_edges = np.histogram_bin_edges(t_all, bins=bins)

            ax.hist(
                [t_all[~sig_arr], t_all[sig_arr]],
                bins=bin_edges,
                stacked=True,
                color=["#d0d0d0", "steelblue"],
                label=["not sig", "sig (FDR)"],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axvline(0, ls="--", color="black", lw=0.8)
            ax.set_xlabel("t-statistic")
            ax.set_ylabel("units")
            ax.set_title(
                f"{name}\n"
                f"n={stats['n']}, sig={stats['n_sig']} "
                f"({stats['frac_sig']:.0%}), "
                f"+{stats['n_pos_sig']}/−{stats['n_neg_sig']}"
            )

            # Annotation with test results
            annot_lines = []
            annot_lines.append(f"Wilcoxon p={stats['wilcoxon_p']:.2g}")
            if n_sig >= 1:
                annot_lines.append(
                    f"Binom +/− p={stats['binom_p']:.2g} "
                    f"({n_pos_sig}/{n_sig})"
                )
            ax.text(
                0.97, 0.95,
                "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=8,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7),
            )

            ax.legend()
            plt.tight_layout()
            if save_path is not None:
                self._save_fig(fig, save_path)
            plt.show()

        return stats

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

    @staticmethod
    def _scatter_with_marginals(
        x, y, categories,
        x_label, y_label, title,
        rho, rho_p,
        figsize,
        sig_x=None,
        sig_y=None,
        fisher_or=None,
        fisher_p=None,
    ):
        """Scatter with marginal histograms, colored by significance category.
 
        Colors are chosen so the "both" color is the additive mix of the
        two single-axis colors:
            x-only  = warm red/orange
            y-only  = blue
            both    = purple  (red + blue)
            neither = grey
 
        Marginal histograms show sig vs not-sig for their own axis only
        (top marginal uses x-axis significance, right marginal uses y-axis
        significance), using a single hue with sig units in saturated color
        and non-sig in a pale tint.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
 
        # --- Color palette: additive logic ---
        #   x-only  = warm red       #e05539
        #   y-only  = steel blue     #4682b4
        #   both    = purple blend   #7b3e9a
        #   neither = light grey     #d0d0d0
        COLOR_X    = "#e05539"
        COLOR_Y    = "#4682b4"
        COLOR_BOTH = "#7b3e9a"
        COLOR_NONE = "#d0d0d0"
 
        # Pale tints for non-sig in marginals
        COLOR_X_PALE = "#f2c4b8"
        COLOR_Y_PALE = "#b8d4e8"
 
        # Build category -> color map (category names are dynamic)
        unique_cats = sorted(set(categories))
        x_only_name = f"{x_label} only"
        y_only_name = f"{y_label} only"
 
        cat_colors = {
            "neither":    COLOR_NONE,
            "both":       COLOR_BOTH,
            x_only_name:  COLOR_X,
            y_only_name:  COLOR_Y,
        }
 
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            4, 4, figure=fig,
            hspace=0.05, wspace=0.05,
        )
 
        ax_main  = fig.add_subplot(gs[1:, :3])
        ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)
 
        # --- Main scatter: draw neither first, then single-axis, then both ---
        draw_order = (
            ["neither"]
            + [c for c in unique_cats if c not in ("neither", "both")]
            + ["both"]
        )
        draw_order = [c for c in draw_order if c in unique_cats]
 
        for cat in draw_order:
            mask = np.array(categories) == cat
            if not mask.any():
                continue
            ax_main.scatter(
                x[mask], y[mask],
                c=cat_colors.get(cat, COLOR_NONE),
                label=f"{cat} ({mask.sum()})",
                s=25, alpha=0.7, edgecolors="none",
            )
 
        ax_main.axhline(0, ls=":", color="grey", lw=0.5)
        ax_main.axvline(0, ls=":", color="grey", lw=0.5)
        ax_main.set_xlabel(f"t  ({x_label})")
        ax_main.set_ylabel(f"t  ({y_label})")
        ax_main.legend(fontsize=7, loc="best")
 
        annot = f"n={len(x)}, ρ={rho:.3f}, p={rho_p:.2g}"
        if fisher_or is not None and fisher_p is not None:
            annot += f"\nFisher OR={fisher_or:.2f}, p={fisher_p:.2g}"
 
        ax_main.text(
            0.02, 0.98,
            annot,
            transform=ax_main.transAxes, fontsize=8,
            va="top", ha="left",
        )
 
        # --- Marginals: per-axis sig/not-sig ---
        bins_edge = np.linspace(
            min(np.nanmin(x), np.nanmin(y)),
            max(np.nanmax(x), np.nanmax(y)),
            31,
        )
 
        # Top marginal: x-axis significance
        if sig_x is not None:
            sig_x = np.asarray(sig_x, dtype=bool)
            ax_top.hist(
                x[~sig_x], bins=bins_edge,
                color=COLOR_X_PALE, alpha=0.8, label="not sig",
            )
            ax_top.hist(
                x[sig_x], bins=bins_edge,
                color=COLOR_X, alpha=0.8, label="sig (FDR)",
            )
        else:
            ax_top.hist(x, bins=bins_edge, color=COLOR_X_PALE, alpha=0.8)
 
        # Right marginal: y-axis significance
        if sig_y is not None:
            sig_y = np.asarray(sig_y, dtype=bool)
            ax_right.hist(
                y[~sig_y], bins=bins_edge, orientation="horizontal",
                color=COLOR_Y_PALE, alpha=0.8, label="not sig",
            )
            ax_right.hist(
                y[sig_y], bins=bins_edge, orientation="horizontal",
                color=COLOR_Y, alpha=0.8, label="sig (FDR)",
            )
        else:
            ax_right.hist(
                y, bins=bins_edge, orientation="horizontal",
                color=COLOR_Y_PALE, alpha=0.8,
            )
 
        ax_top.set_ylabel("count")
        ax_right.set_xlabel("count")
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
 
        fig.suptitle(title, fontsize=11, y=0.98)
 
        return fig, ax_main, ax_top, ax_right

    @staticmethod
    def _save_fig(fig, path: str, dpi: int = 300):
        """Save figure as both .png and .svg to the given path (without extension)."""
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        stem = p.parent / p.stem  # strip any extension the user included
        fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{stem}.svg", bbox_inches="tight")
        print(f"Saved: {stem}.png, {stem}.svg")
