"""
spatial_encoding.py — Spatial topography of LC encoding t-statistics.

Plug-in for PerUnitStatsRegistry.

    from spatial_encoding import SpatialEncoder

    enc = SpatialEncoder(filtered_ephys, mesh_path=MESH_PATH)

    # 3-panel anatomical map (sag | hor | cor) colored by t-stat
    fig = enc.plot(reg, "ols_rt_response")

    # permutation-based spatial dependence test
    res = enc.test(reg, "ols_rt_response")


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from trimesh import load_mesh as _load_mesh
except ImportError:
    _load_mesh = None

try:
    from ccf_utils import pir_to_lps, project_to_plane
except ImportError:
    pir_to_lps = None
    project_to_plane = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ML, _AP, _DV = 0, 1, 2
_BREGMA_LPS_MM = np.array([-5.7, 5.4, -0.45])
_BREGMA_PIR_VOX = np.array([216, 18, 228])
_CCF_RES_UM = 25.0
_PLANES = {"sag": [_AP, _DV], "hor": [_ML, _AP], "cor": [_ML, _DV]}


# ---------------------------------------------------------------------------
# Spatial dependence test
# ---------------------------------------------------------------------------
def spatial_dependence_summary(
    coords,
    values,
    *,
    k_neighbors=15,
    n_splits=5,
    permutations=5000,
    seed=0,
    return_null=False,
):
    """
    Test whether *values* vary with 3-D anatomical *coords*.

    Two permutation-based tests:
      1. Linear trend: value ~ x + y + z  -> R^2 permutation p
      2. kNN CV predictability             -> R^2_cv permutation p

    Returns
    -------
    dict with keys 'linear_trend', 'cv_predictability_knn', 'n_used'.
    """
    import statsmodels.api as sm
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score

    X = np.asarray(coords, float).reshape(len(coords), -1)
    y = np.asarray(values, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    n = len(y)
    if n < 10:
        raise ValueError(f"Too few valid points (n={n}).")

    def _seeds(offset):
        ss = np.random.SeedSequence(seed + offset).spawn(permutations)
        return np.array(
            [s.generate_state(1, dtype=np.uint32)[0] for s in ss], dtype=np.uint32
        )

    # (1) linear trend
    X_c = sm.add_constant(X)
    model = sm.OLS(y, X_c).fit()
    r2_obs = float(model.rsquared)

    seeds_lin = _seeds(12345)
    r2_null = np.array(
        [sm.OLS(np.random.default_rng(int(s)).permutation(y), X_c).fit().rsquared
         for s in seeds_lin]
    )
    p_trend = (np.sum(r2_null >= r2_obs) + 1) / (permutations + 1)

    # (2) kNN CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _cv_r2(yy):
        preds = np.zeros_like(yy)
        for tr, te in kf.split(X):
            knn = KNeighborsRegressor(
                n_neighbors=min(k_neighbors, len(tr)), weights="distance"
            )
            knn.fit(X[tr], yy[tr])
            preds[te] = knn.predict(X[te])
        return r2_score(yy, preds)

    r2_cv_obs = _cv_r2(y)
    seeds_knn = _seeds(0)
    r2_cv_null = np.array(
        [_cv_r2(np.random.default_rng(int(s)).permutation(y)) for s in seeds_knn]
    )
    p_cv = (np.sum(r2_cv_null >= r2_cv_obs) + 1) / (permutations + 1)

    out = {
        "n_used": int(n),
        "linear_trend": {
            "coef": model.params.tolist(),
            "r2": r2_obs,
            "p_perm": float(p_trend),
        },
        "cv_predictability_knn": {
            "r2_cv": float(r2_cv_obs),
            "p_perm": float(p_cv),
        },
    }
    if return_null:
        out["linear_trend"]["null"] = r2_null.tolist()
        out["cv_predictability_knn"]["null"] = r2_cv_null.tolist()
    return out


# ---------------------------------------------------------------------------
# SpatialEncoder
# ---------------------------------------------------------------------------
class SpatialEncoder:
    """
    Spatial topography for PerUnitStatsRegistry entries.

    Parameters
    ----------
    features_df : DataFrame
        Unit table with x_ccf, y_ccf, z_ccf, session_prefix, unit.
    mesh_path : str or None
        LC mesh .obj for contour backgrounds.
    fold_left : bool
        Mirror ML to left hemisphere.
    """

    def __init__(self, features_df, mesh_path=None, fold_left=True):
        self.features_df = features_df.copy()
        self.fold_left = fold_left
        self._contours = {}
        if mesh_path and _load_mesh and pir_to_lps and project_to_plane:
            self._load_mesh(mesh_path)

    # def _load_mesh(self, path):
    #     mesh = _load_mesh(path)
    #     v = np.array(mesh.vertices, float)
    #     v = (v - _BREGMA_PIR_VOX) * (_CCF_RES_UM / 1000.0)
    #     v = pir_to_lps(v)
    #     self._contours = {
    #         name: project_to_plane(v, axes, pitch=0.02, margin=0.5)
    #         for name, axes in _PLANES.items()
    #     }
    def _load_mesh(self, path):
        mesh = _load_mesh(path)
        v = np.array(mesh.vertices, float) / 1000.0       # µm → mm
        v[:, 2] = -v[:, 2]                                 # negate Z
        v[:, 0] = -v[:, 0]                                 # negate X
        v = v - _BREGMA_LPS_MM                             # bregma-center
        if self.fold_left:
            v[:, _ML] = np.abs(v[:, _ML])                  # positive ML
        self._contours = {
            name: project_to_plane(v, axes, pitch=0.02, margin=0.5)
            for name, axes in _PLANES.items()
        }
    

    # ---- join registry t-stats to CCF coords ----

    @staticmethod
    def _canon_unit(x):
        """Normalize unit id to str(int) — matches registry convention."""
        try:
            return str(int(float(x)))
        except Exception:
            return str(x)

    def _join_entry(self, reg, entry_name):
        """
        Inner-join registry entry to features_df on (session_prefix, unit).
        Returns (coords_N3, t_values_N, sig_N_bool).
        """
        entry_df = reg.get(entry_name)
        feat = self.features_df.copy()
        feat["unit"] = feat["unit"].map(self._canon_unit)
        entry_df["unit"] = entry_df["unit"].map(self._canon_unit)
        merged = feat.merge(
            entry_df[["session_prefix", "unit", "t", "sig_fdr"]],
            on=["session_prefix", "unit"],
            how="inner",
        )
        coords = merged[["x_ccf", "y_ccf", "z_ccf"]].to_numpy(float) - _BREGMA_LPS_MM
        if self.fold_left:
            coords[:, _ML] = np.abs(coords[:, _ML]) #positive
        return coords, merged["t"].to_numpy(float), merged["sig_fdr"].to_numpy(bool)

    # ---- statistical testing ----

    def test(self, reg, entry_name, *, permutations=5000, k_neighbors=10,
             n_splits=5, seed=0, also_abs=True, return_null=False):
        """
        Spatial dependence test on signed t-stats (and optionally |t|).

        Returns dict with 'signed' and optionally 'abs' sub-dicts.
        """
        coords, vals, _ = self._join_entry(reg, entry_name)
        out = {
            "entry": entry_name,
            "signed": spatial_dependence_summary(
                coords, vals,
                k_neighbors=k_neighbors, n_splits=n_splits,
                permutations=permutations, seed=seed, return_null=return_null,
            ),
        }
        if also_abs:
            out["abs"] = spatial_dependence_summary(
                coords, np.abs(vals),
                k_neighbors=k_neighbors, n_splits=n_splits,
                permutations=permutations, seed=seed, return_null=return_null,
            )
        return out

    # ---- plotting ----

    def plot(
        self,
        reg,
        entry_name,
        *,
        abs_value=False,
        use_sig=False,
        point_size=15,
        alpha=0.7,
        dv_ylim=(-5, -3),
        title=None,
        save_path=None,
        cmap=None,
    ):
        """
        3-panel anatomical map: sagittal | horizontal | coronal.
        Points colored by t-stat with colorbar.

        Parameters
        ----------
        abs_value : bool
            If True, plot |t| with a sequential colormap (viridis_r).
        use_sig : bool
            If True, non-significant units are greyed out.
        save_path : str or None
            Saves .png + .svg (pass without extension).
        """
        coords, vals, sig = self._join_entry(reg, entry_name)
        if abs_value:
            vals = np.abs(vals)
        base = np.isfinite(vals)

        if not np.any(base):
            raise ValueError(f"No finite t-stats for '{entry_name}'")

        if use_sig:
            sig_mask = sig & base
            nonsig_mask = ~sig & base
        else:
            sig_mask = base
            nonsig_mask = np.zeros_like(base, dtype=bool)

        # norm + cmap: diverging for signed, sequential for abs
        ref = vals[sig_mask] if np.any(sig_mask) else vals[base]
        ref = ref[np.isfinite(ref)]

        if abs_value:
            cmap = cmap or "viridis_r"
            amp = np.nanquantile(ref, 0.92)
            amp = max(amp if np.isfinite(amp) else 1.0, 1e-6)
            norm = Normalize(vmin=0, vmax=amp)
        else:
            cmap = cmap or "coolwarm"
            amp = np.nanquantile(np.abs(ref), 0.92)
            amp = max(amp if np.isfinite(amp) else 1.0, 1e-6)
            norm = Normalize(vmin=-amp, vmax=amp)

        cm = plt.get_cmap(cmap)

        rgba_sig = cm(norm(vals[sig_mask]))
        rgba_sig[:, -1] = alpha
        coords_sig = coords[sig_mask]
        coords_non = coords[nonsig_mask]

        _ax_label = {
            _ML: "ML (mm, folded)" if self.fold_left else "ML (mm)",
            _AP: "AP (mm)",
            _DV: "DV (mm)",
        }

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        for (pname, paxes), ax in zip(_PLANES.items(), axes):
            for c in self._contours.get(pname, []):
                cp = c.copy()
                ax.fill(cp[:, 0], cp[:, 1], color="lightgray", alpha=0.3, lw=0)

            if np.any(nonsig_mask):
                ax.scatter(
                    coords_non[:, paxes[0]], coords_non[:, paxes[1]],
                    color="lightgrey", s=point_size, alpha=0.5, edgecolors="none",
                )

            ax.scatter(
                coords_sig[:, paxes[0]], coords_sig[:, paxes[1]],
                facecolors=rgba_sig, edgecolors="none", s=point_size,
            )

            ax.set_xlabel(_ax_label[paxes[0]])
            ax.set_ylabel(_ax_label[paxes[1]])
            ax.set_aspect("equal")
            if dv_ylim and paxes[1] == _DV:
                ax.set_ylim(dv_ylim)

        # colorbar on coronal panel
        sm = ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        div = make_axes_locatable(axes[-1])
        cax = div.append_axes("right", size="5%", pad="3%")
        fig.colorbar(sm, cax=cax, label="|t|" if abs_value else "t-stat")

        fig.suptitle(title or entry_name, fontsize=11)
        plt.tight_layout()

        if save_path:
            from pathlib import Path
            stem = Path(save_path)
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".png", ".svg"):
                fig.savefig(str(stem) + ext, dpi=300, bbox_inches="tight")

        return fig, axes

    def plot_subgroups(
        self,
        reg,
        entry_name,
        *,
        sig_only=True,
        point_size=20,
        alpha=0.8,
        dv_ylim=(-5, -3),
        colors=None,
        title=None,
        save_path=None,
        permutations=5000,
        seed=0,
    ):
        """
        Split units into neg / pos by t-stat sign, plot on 3-panel map,
        and test whether each subgroup is more spatially clustered than
        expected by chance (sampling from the full population).

        Clustering statistic: mean nearest-neighbor distance among the
        k subgroup members, compared to the null of randomly drawing k
        units from all recorded units.

        Parameters
        ----------
        sig_only : bool
            If True, only FDR-significant units form the neg/pos groups.
        permutations : int
            Null distribution size for clustering test.

        Returns
        -------
        fig, axes, results : dict
            results['neg'] and results['pos'] each contain n, mean_nnd,
            p_clustering (one-sided: is subgroup more clustered than chance?).
        """
        from scipy.spatial.distance import pdist, squareform

        coords, vals, sig = self._join_entry(reg, entry_name)
        base = np.isfinite(vals) & np.all(np.isfinite(coords), axis=1)
        colors = colors or {"neg": "#4682b4", "pos": "#e05539", "nonsig": "#d0d0d0"}

        if sig_only:
            neg_mask = base & sig & (vals < 0)
            pos_mask = base & sig & (vals > 0)
            other_mask = base & ~(sig & (vals != 0))
        else:
            neg_mask = base & (vals < 0)
            pos_mask = base & (vals > 0)
            other_mask = np.zeros_like(base, dtype=bool)

        n_neg, n_pos = int(neg_mask.sum()), int(pos_mask.sum())

        _ax_label = {
            _ML: "ML (mm)" if self.fold_left else "ML (mm)",
            _AP: "AP (mm)",
            _DV: "DV (mm)",
        }

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        for (pname, paxes), ax in zip(_PLANES.items(), axes):
            for c in self._contours.get(pname, []):
                cp = c.copy()
                if self.fold_left:
                    if paxes[0] == _ML:
                        cp[:, 0] *= -1
                    if paxes[1] == _ML:
                        cp[:, 1] *= -1
                ax.fill(cp[:, 0], cp[:, 1], color="lightgray", alpha=0.3, lw=0)

            if np.any(other_mask):
                ax.scatter(
                    coords[other_mask, paxes[0]], coords[other_mask, paxes[1]],
                    color=colors["nonsig"], s=point_size * 0.6, alpha=0.4,
                    edgecolors="none", label=f"nonsig ({int(other_mask.sum())})",
                )
            if n_neg > 0:
                ax.scatter(
                    coords[neg_mask, paxes[0]], coords[neg_mask, paxes[1]],
                    color=colors["neg"], s=point_size, alpha=alpha,
                    edgecolors="white", linewidths=0.5,
                    label=f"neg ({n_neg})",
                )
            if n_pos > 0:
                ax.scatter(
                    coords[pos_mask, paxes[0]], coords[pos_mask, paxes[1]],
                    color=colors["pos"], s=point_size, alpha=alpha,
                    edgecolors="white", linewidths=0.5,
                    label=f"pos ({n_pos})",
                )

            ax.set_xlabel(_ax_label[paxes[0]])
            ax.set_ylabel(_ax_label[paxes[1]])
            ax.set_aspect("equal")
            if dv_ylim and paxes[1] == _DV:
                ax.set_ylim(dv_ylim)

        axes[1].legend(fontsize=7, loc="best")

        # --- clustering test per subgroup ---
        # pool = all units with valid coords (the reference population)
        pool_coords = coords[base]
        n_pool = len(pool_coords)

        # precompute full pairwise distance matrix once
        if n_pool > 1:
            D_full = squareform(pdist(pool_coords))
            np.fill_diagonal(D_full, np.inf)  # exclude self

        def _mean_nnd(idx):
            """Mean nearest-neighbor distance for a subset (indices into pool)."""
            if len(idx) < 2:
                return np.nan
            D_sub = D_full[np.ix_(idx, idx)]
            return float(np.mean(np.min(D_sub, axis=1)))

        rng = np.random.default_rng(seed)

        def _test_group(mask, label):
            # map subgroup mask -> indices into pool
            pool_idx = np.where(base)[0]
            group_idx_in_pool = np.array(
                [np.searchsorted(pool_idx, i) for i in np.where(mask)[0]]
            )
            k = len(group_idx_in_pool)
            if k < 2:
                return {"n": k, "mean_nnd": np.nan, "p_clustering": np.nan}

            obs_nnd = _mean_nnd(group_idx_in_pool)

            # null: draw k random units from pool
            null_nnds = np.empty(permutations)
            for i in range(permutations):
                rand_idx = rng.choice(n_pool, size=k, replace=False)
                null_nnds[i] = _mean_nnd(rand_idx)

            # one-sided: is subgroup MORE clustered (smaller NND) than chance?
            p_clust = (np.sum(null_nnds <= obs_nnd) + 1) / (permutations + 1)

            return {
                "n": k,
                "mean_nnd": obs_nnd,
                "p_clustering": float(p_clust),
            }

        results = {
            "neg": _test_group(neg_mask, "neg"),
            "pos": _test_group(pos_mask, "pos"),
        }

        # title with stats
        parts = [title or entry_name]
        for label in ("neg", "pos"):
            r = results[label]
            if np.isfinite(r.get("p_clustering", np.nan)):
                parts.append(
                    f"{label}: n={r['n']}, NND={r['mean_nnd']:.3f}mm, "
                    f"p={r['p_clustering']:.4f}"
                )
        fig.suptitle("\n".join(parts), fontsize=9)
        plt.tight_layout()

        if save_path:
            from pathlib import Path
            stem = Path(save_path)
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".png", ".svg"):
                fig.savefig(str(stem) + ext, dpi=300, bbox_inches="tight")

        return fig, axes, results

    def plot_compare(
        self,
        reg,
        name_a,
        name_b,
        *,
        merged=None,
        point_size=20,
        alpha=0.8,
        dv_ylim=(-5, -3),
        title=None,
        save_path=None,
    ):
        """
        Plot the 4 significance categories from reg.compare() on the
        3-panel anatomical map.

        Parameters
        ----------
        merged : DataFrame or None
            Output of reg.compare(name_a, name_b, show=False).
            If None, calls compare internally.
        """
        if merged is None:
            merged = reg.compare(name_a, name_b, show=False)

        # canonicalize unit keys for join
        feat = self.features_df.copy()
        feat["unit"] = feat["unit"].map(self._canon_unit)
        merged = merged.copy()
        merged["unit"] = merged["unit"].map(self._canon_unit)

        joined = feat.merge(
            merged[["session_prefix", "unit", "sig_category"]],
            on=["session_prefix", "unit"],
            how="inner",
        )

        coords = joined[["x_ccf", "y_ccf", "z_ccf"]].to_numpy(float) - _BREGMA_LPS_MM
        if self.fold_left:
            coords[:, _ML] = -np.abs(coords[:, _ML])

        cats = joined["sig_category"].values
        valid = np.all(np.isfinite(coords), axis=1)

        # color map matching registry compare colors
        a_only = f"{name_a} only"
        b_only = f"{name_b} only"
        cat_colors = {
            "neither":  "#d0d0d0",
            a_only:     "#e05539",
            b_only:     "#4682b4",
            "both":     "#7b3e9a",
        }
        # draw order: neither first, both last
        draw_order = ["neither", a_only, b_only, "both"]

        _ax_label = {
            _ML: "ML (mm, folded)" if self.fold_left else "ML (mm)",
            _AP: "AP (mm)",
            _DV: "DV (mm)",
        }

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        for (pname, paxes), ax in zip(_PLANES.items(), axes):
            for c in self._contours.get(pname, []):
                cp = c.copy()
                if self.fold_left:
                    if paxes[0] == _ML:
                        cp[:, 0] *= -1
                    if paxes[1] == _ML:
                        cp[:, 1] *= -1
                ax.fill(cp[:, 0], cp[:, 1], color="lightgray", alpha=0.3, lw=0)

            for cat in draw_order:
                mask = valid & (cats == cat)
                n_cat = int(mask.sum())
                if n_cat == 0:
                    continue
                is_nonsig = (cat == "neither")
                ax.scatter(
                    coords[mask, paxes[0]], coords[mask, paxes[1]],
                    color=cat_colors.get(cat, "#d0d0d0"),
                    s=point_size * (0.6 if is_nonsig else 1.0),
                    alpha=0.4 if is_nonsig else alpha,
                    edgecolors="none" if is_nonsig else "white",
                    linewidths=0 if is_nonsig else 0.5,
                    label=f"{cat} ({n_cat})",
                )

            ax.set_xlabel(_ax_label[paxes[0]])
            ax.set_ylabel(_ax_label[paxes[1]])
            ax.set_aspect("equal")
            if dv_ylim and paxes[1] == _DV:
                ax.set_ylim(dv_ylim)

        # legend outside, to the right of the last panel
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, loc="center right",
                   bbox_to_anchor=(1.15, 0.5), frameon=False)
        fig.suptitle(title or f"{name_a} vs {name_b}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        
        if save_path:
            from pathlib import Path
            stem = Path(save_path)
            stem.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".png", ".svg"):
                fig.savefig(str(stem) + ext, dpi=300, bbox_inches="tight")

        return fig, axes