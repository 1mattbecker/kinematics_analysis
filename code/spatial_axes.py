"""
spatial_axes.py — Fitting and comparing 3D spatial *axes* (gradient directions).

Given per-unit (or per-cell) CCF coordinates and some feature measured at each
location, find the single 3D direction along which that feature varies most,
bootstrap its uncertainty, and test whether two such directions differ.

    from spatial_axes import (
        bootstrap_spatial_axis_linear, bootstrap_spatial_axis_cca,
        compare_bootstrap_directions, cone_half_angle,
    )

    res  = bootstrap_spatial_axis_linear(t_stats, coords_lps_mm, n_boot=2000, seed=42)
    axis = res["axis_unit"]                              # (3,) unit vector
    half_angle, _ = cone_half_angle(axis, res["axis_boot"], q=95)
    cmp  = compare_bootstrap_directions(axis_a, axis_b, boot_a, boot_b)

Three fitting methods, chosen by feature type:

===========  ==========================  ===================================
Method       Feature                     Example
===========  ==========================  ===================================
``linear``   scalar per location (OLS)   RT-encoding t-statistic per unit
``cca``      multivariate per location   waveform features, MERFISH genes
``LDA``      categorical per location    retrograde injection region
===========  ==========================  ===================================

Each ``fit_*`` returns the observed axis; each ``bootstrap_*`` wrapper resamples
locations with replacement and hemisphere-aligns every resample to the observed
axis (an axis has no intrinsic sign, so ``v`` and ``-v`` describe the same
gradient — without alignment the bootstrap cloud is bimodal and its spread is
meaningless).

Relationship to ``spatial_encoding.py``
---------------------------------------
Different question, different inputs. ``spatial_encoding.SpatialEncoder`` asks
*where* in CCF space a statistic is large (anatomical maps, subgroup maps,
spatial-dependence permutation tests) and contains no axis-fitting machinery.
This module asks *in which direction* it changes. Keep them separate.

Sign conventions
----------------
A fitted axis direction is arbitrary up to sign. ``bootstrap_*`` fixes the sign
*within* a run (all resamples aligned to the observed axis), but not *across*
runs or *between* axes. Callers that plot several axes together should adopt an
explicit convention — e.g. ``eph_09`` flips the retrograde LDA axis to have a
positive dot product with the waveform axis — or the figure will flip between
runs.

Provenance
----------
Ported from ``spatial_axis_comparison_rt_encoding_update.ipynb`` cells 14, 15
and 31, which adapted Han Yu's ``F_spatial-axis-comparison.ipynb``. Function
bodies are unchanged from that source; ``eph_08_waveform_axis.ipynb`` previously
carried private inline copies of the two CCA functions and now imports them
from here.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# ── Core ──────────────────────────────────────────────────────────────────────

def _unit(v, eps: float = 1e-15) -> np.ndarray:
    """Normalize a vector to unit length.

    Parameters
    ----------
    v : array-like
        Vector to normalize; flattened to 1D.
    eps : float
        Norms below this are treated as zero.

    Returns
    -------
    numpy.ndarray
        Unit-norm copy of ``v``.

    Raises
    ------
    ValueError
        If ``v`` has near-zero norm and no direction can be defined.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


# ── Linear: scalar feature ~ 3D coords ────────────────────────────────────────

def fit_spatial_axis_linear(feature, coords, *, add_intercept: bool = True) -> Dict:
    """Fit a spatial gradient axis for a scalar feature by OLS.

    Regresses ``feature ~ 1 + coords`` and takes the direction of the
    coordinate coefficients as the axis along which the feature changes fastest.

    Parameters
    ----------
    feature : array-like, shape (n,)
        Scalar value per location (e.g. a per-unit RT-encoding t-statistic).
    coords : array-like, shape (n, 3)
        3D coordinates, one row per location (bregma-centered LPS mm).
    add_intercept : bool
        Include an intercept column in the design matrix.

    Returns
    -------
    dict
        ``axis_unit`` (3,) unit-vector direction, ``beta`` (3,) raw slope
        coefficients, ``intercept`` (kept at 0.0 for a stable return shape).

    Raises
    ------
    ValueError
        If fewer than 4 rows are finite in both ``feature`` and ``coords``.
    """
    y = np.asarray(feature, dtype=float).reshape(-1)
    X = np.asarray(coords, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 4:
        raise ValueError("Need >= 4 valid samples.")

    if add_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
        coef = np.linalg.lstsq(X_design, y, rcond=None)[0]
        beta = coef[1:]
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    beta = np.asarray(beta, dtype=float).reshape(3)
    return {"axis_unit": _unit(beta), "beta": beta, "intercept": 0.0}


def bootstrap_spatial_axis_linear(
    feature, coords, *, n_boot: int = 2000, seed: int = 0,
    align_to_observed: bool = True,
) -> Dict:
    """Bootstrap the linear spatial axis direction.

    Resamples locations with replacement and refits, so the spread of
    ``axis_boot`` measures how well the gradient direction is determined.

    Parameters
    ----------
    feature : array-like, shape (n,)
        Scalar value per location.
    coords : array-like, shape (n, 3)
        3D coordinates, one row per location.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Seed for ``numpy.random.default_rng``.
    align_to_observed : bool
        Flip each resampled axis to the observed hemisphere. Leave True unless
        you specifically want the unaligned (bimodal) cloud.

    Returns
    -------
    dict
        ``axis_unit`` and ``beta`` from the full-sample fit, plus ``axis_boot``
        (n_valid, 3), ``n_boot_valid`` and ``n_boot_failed``.
    """
    y = np.asarray(feature, dtype=float).reshape(-1)
    X = np.asarray(coords, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n = len(y)

    obs = fit_spatial_axis_linear(y, X)
    axis_obs = obs["axis_unit"]
    rng = np.random.default_rng(seed)

    axis_boot = []
    failed = 0
    for _ in range(n_boot):
        ind = rng.integers(0, n, size=n)
        try:
            res_b = fit_spatial_axis_linear(y[ind], X[ind])
            axis_b = res_b["axis_unit"]
            if align_to_observed and np.dot(axis_b, axis_obs) < 0:
                axis_b = -axis_b
            axis_boot.append(axis_b)
        except Exception:
            failed += 1

    return {
        "axis_unit": axis_obs,
        "beta": obs["beta"],
        "axis_boot": np.array(axis_boot),
        "n_boot_valid": len(axis_boot),
        "n_boot_failed": failed,
    }


# ── CCA: multivariate features ~ 3D coords ────────────────────────────────────

def fit_spatial_axis_cca(features, coords, *, standardize_features: bool = True) -> Dict:
    """Fit a spatial axis for multivariate features by canonical correlation.

    Finds the 3D direction whose projection is maximally correlated with some
    linear combination of the features — the spatial axis along which the
    feature *profile* changes most.

    Parameters
    ----------
    features : array-like, shape (n, k) or (n,)
        Feature matrix, one row per location. 1D input is treated as k=1.
        Zero-variance columns are dropped.
    coords : array-like, shape (n, 3)
        3D coordinates, one row per location.
    standardize_features : bool
        Z-score features before fitting, so features on different scales
        contribute comparably.

    Returns
    -------
    dict
        ``axis_unit`` (3,), ``beta`` (3,) raw CCA x-weights, ``intercept``
        (0.0), and ``canonical_corr``, the first canonical correlation.

    Raises
    ------
    ValueError
        If fewer than 4 rows are finite in both ``features`` and ``coords``.
    """
    S = np.asarray(features, dtype=float)
    X = np.asarray(coords, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(S), axis=1)
    X, S = X[ok], S[ok]
    if len(X) < 4:
        raise ValueError("Need >= 4 valid samples.")

    keep_feat = np.nanstd(S, axis=0) > 0
    S = S[:, keep_feat]
    if standardize_features:
        S = StandardScaler().fit_transform(S)

    max_comp = min(1, X.shape[1], S.shape[1], len(X) - 1)
    cca = CCA(n_components=max_comp)
    cca.fit(X, S)
    X_c, S_c = cca.transform(X, S)
    canonical_corr = float(np.corrcoef(X_c[:, 0], S_c[:, 0])[0, 1])

    beta = np.asarray(cca.x_weights_[:, 0], dtype=float).reshape(3)
    return {"axis_unit": _unit(beta), "beta": beta, "intercept": 0.0,
            "canonical_corr": canonical_corr}


def bootstrap_spatial_axis_cca(
    features, coords, *, n_boot: int = 2000, seed: int = 0,
    align_to_observed: bool = True, standardize_features: bool = True,
) -> Dict:
    """Bootstrap the CCA spatial axis direction.

    Parameters
    ----------
    features : array-like, shape (n, k) or (n,)
        Feature matrix, one row per location.
    coords : array-like, shape (n, 3)
        3D coordinates, one row per location.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Seed for ``numpy.random.default_rng``.
    align_to_observed : bool
        Flip each resampled axis to the observed hemisphere.
    standardize_features : bool
        Z-score features before each fit.

    Returns
    -------
    dict
        ``axis_unit``, ``beta`` and ``canonical_corr`` from the full-sample fit,
        plus ``axis_boot`` (n_valid, 3), ``n_boot_valid`` and ``n_boot_failed``.
    """
    S = np.asarray(features, dtype=float)
    X = np.asarray(coords, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(S), axis=1)
    X, S = X[ok], S[ok]

    obs = fit_spatial_axis_cca(S, X, standardize_features=standardize_features)
    axis_obs = obs["axis_unit"]
    rng = np.random.default_rng(seed)
    n = len(X)

    axis_boot = []
    failed = 0
    for _ in range(n_boot):
        ind = rng.integers(0, n, size=n)
        try:
            res_b = fit_spatial_axis_cca(S[ind], X[ind], standardize_features=standardize_features)
            axis_b = res_b["axis_unit"]
            if align_to_observed and np.dot(axis_b, axis_obs) < 0:
                axis_b = -axis_b
            axis_boot.append(axis_b)
        except Exception:
            failed += 1

    return {
        "axis_unit": axis_obs, "beta": obs["beta"],
        "canonical_corr": obs["canonical_corr"],
        "axis_boot": np.array(axis_boot),
        "n_boot_valid": len(axis_boot), "n_boot_failed": failed,
    }


# ── LDA: categorical labels ~ 3D coords ───────────────────────────────────────

def _pd_notnull_1d(arr) -> np.ndarray:
    """Boolean mask of non-null entries in a 1D object array.

    Stands in for ``pandas.notnull`` so the module stays pandas-free; handles
    ``None`` and float ``nan`` inside an object-dtype array of string labels.

    Parameters
    ----------
    arr : array-like
        1D array of labels, typically object dtype.

    Returns
    -------
    numpy.ndarray of bool
        True where the entry is not None and not NaN.
    """
    arr = np.asarray(arr, dtype=object)
    out = np.ones(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            out[i] = False
        else:
            try:
                if isinstance(v, float) and np.isnan(v):
                    out[i] = False
            except Exception:
                pass
    return out


def fit_spatial_axis_LDA(labels, coords) -> Dict:
    """Fit a spatial axis separating categorical groups by linear discriminant.

    The first discriminant direction is the 3D axis along which the labelled
    groups are best separated (e.g. LC cells by retrograde injection region).

    Parameters
    ----------
    labels : array-like, shape (n,)
        Categorical label per location. None/NaN entries are dropped.
    coords : array-like, shape (n, 3)
        3D coordinates, one row per location.

    Returns
    -------
    dict
        ``axis_unit`` (3,), ``beta`` (3,) raw LDA scalings, ``intercept`` (0.0).

    Raises
    ------
    ValueError
        If fewer than 2 classes survive, or any class has fewer than 2 members.

    Notes
    -----
    The sign of a discriminant axis is arbitrary — it depends on class ordering.
    Align it to a reference axis if you plot it alongside others.
    """
    y = np.asarray(labels)
    X = np.asarray(coords, dtype=float)
    ok = np.all(np.isfinite(X), axis=1) & _pd_notnull_1d(y)
    y, X = y[ok], X[ok]

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("Need >= 2 classes with >= 2 samples each.")

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    beta = np.asarray(lda.scalings_[:, 0], dtype=float).reshape(3)
    return {"axis_unit": _unit(beta), "beta": beta, "intercept": 0.0}


def bootstrap_spatial_axis_LDA(
    labels, coords, *, n_boot: int = 2000, seed: int = 0,
    align_to_observed: bool = True,
) -> Dict:
    """Bootstrap the LDA spatial axis direction.

    Resamples that lose a class (or drop one below 2 members) are counted as
    failures rather than fitted, so ``n_boot_failed`` is informative for
    imbalanced label sets.

    Parameters
    ----------
    labels : array-like, shape (n,)
        Categorical label per location.
    coords : array-like, shape (n, 3)
        3D coordinates, one row per location.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Seed for ``numpy.random.default_rng``.
    align_to_observed : bool
        Flip each resampled axis to the observed hemisphere.

    Returns
    -------
    dict
        ``axis_unit`` and ``beta`` from the full-sample fit, plus ``axis_boot``
        (n_valid, 3), ``n_boot_valid`` and ``n_boot_failed``.
    """
    y = np.asarray(labels)
    X = np.asarray(coords, dtype=float)
    ok = np.all(np.isfinite(X), axis=1) & _pd_notnull_1d(y)
    y, X = y[ok], X[ok]
    n = len(y)

    obs = fit_spatial_axis_LDA(y, X)
    axis_obs = obs["axis_unit"]
    rng = np.random.default_rng(seed)

    axis_boot = []
    failed = 0
    for _ in range(n_boot):
        ind = rng.integers(0, n, size=n)
        yb, Xb = y[ind], X[ind]
        classes_b, counts_b = np.unique(yb, return_counts=True)
        if len(classes_b) < 2 or np.any(counts_b < 2):
            failed += 1
            continue
        try:
            res_b = fit_spatial_axis_LDA(yb, Xb)
            axis_b = res_b["axis_unit"]
            if align_to_observed and np.dot(axis_b, axis_obs) < 0:
                axis_b = -axis_b
            axis_boot.append(axis_b)
        except Exception:
            failed += 1

    return {
        "axis_unit": axis_obs, "beta": obs["beta"],
        "axis_boot": np.array(axis_boot),
        "n_boot_valid": len(axis_boot), "n_boot_failed": failed,
    }


# ── Direction comparison (bootstrap Wald test) ────────────────────────────────

def _orthonormal_basis_perp(v0, v1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal tangent basis at ``v0``, with e1 pointing toward ``v1``.

    Directions live on the unit sphere, so differences between them are
    compared in the 2D tangent plane at the reference direction.

    Parameters
    ----------
    v0 : array-like, shape (3,)
        Reference direction; becomes the normal of the tangent plane.
    v1 : array-like, shape (3,)
        Direction used to orient ``e1``. If it is (anti)parallel to ``v0``, an
        arbitrary perpendicular is chosen instead.

    Returns
    -------
    e1, e2, u0 : numpy.ndarray
        Two orthonormal tangent vectors and the normalized reference direction.
    """
    u0 = _unit(v0)
    diff = v1 - np.dot(v1, u0) * u0
    if np.linalg.norm(diff) < 1e-12:
        a = np.array([1., 0., 0.]) if abs(u0[0]) < 0.9 else np.array([0., 1., 0.])
        diff = a - np.dot(a, u0) * u0
    e1 = _unit(diff)
    e2 = _unit(np.cross(u0, e1))
    return e1, e2, u0


def compare_bootstrap_directions(b_x, b_y, b_x_boot, b_y_boot) -> Dict:
    """Test whether two 3D unit-vector directions are significantly different.

    Both axes and their bootstrap clouds are projected into the 2D tangent
    plane at ``b_x``. The paired bootstrap difference gives a covariance, and a
    Wald statistic asks whether the observed offset of ``b_y`` from ``b_x`` is
    large relative to that bootstrap spread.

    Parameters
    ----------
    b_x, b_y : array-like, shape (3,)
        The two observed axis directions to compare.
    b_x_boot, b_y_boot : array-like, shape (n_boot, 3)
        Bootstrap clouds for each axis. Rows are paired by position, so pass
        equal-length slices of the two clouds (the shorter length is used).
        Both are hemisphere-aligned and renormalized in place of the caller's
        intent, on local copies.

    Returns
    -------
    dict
        ``angle_deg`` (observed angle between the axes), ``W_obs`` (Wald
        statistic), ``p_chi2`` (χ², df=2), ``p_boot`` (bootstrap p-value),
        plus ``d_obs``, ``d_boot``, ``mean_boot`` and ``cov_2d`` for plotting.

    Notes
    -----
    Angle and p-value answer different questions and should be read together:
    a small angle with a significant p means two axes that are close but
    distinguishable given tight bootstraps, while a large angle with a
    non-significant p means the data cannot pin either direction down.
    """
    b_x, b_y = _unit(b_x), _unit(b_y)
    b_x_boot = np.array(b_x_boot, dtype=float, copy=True)
    b_y_boot = np.array(b_y_boot, dtype=float, copy=True)

    # Hemisphere-align
    flip_x = np.sum(b_x_boot * b_x[None, :], axis=1) < 0
    flip_y = np.sum(b_y_boot * b_y[None, :], axis=1) < 0
    b_x_boot[flip_x] *= -1
    b_y_boot[flip_y] *= -1

    # Normalize
    b_x_boot = b_x_boot / np.linalg.norm(b_x_boot, axis=1, keepdims=True)
    b_y_boot = b_y_boot / np.linalg.norm(b_y_boot, axis=1, keepdims=True)

    # Tangent plane at b_x, e1 aligned toward b_y
    e1, e2, u0 = _orthonormal_basis_perp(b_x, b_y)
    A = np.vstack([e1, e2])  # 2x3

    d_obs = A @ b_y
    Px = (A @ b_x_boot.T).T
    Py = (A @ b_y_boot.T).T

    n_boot = min(len(Px), len(Py))
    d_boot = Py[:n_boot] - Px[:n_boot]
    mean_boot = d_boot.mean(axis=0)

    cov_2d = np.cov(d_boot.T, ddof=1) + np.eye(2) * 1e-12
    cov_inv = np.linalg.inv(cov_2d)

    W_obs = float(d_obs.T @ cov_inv @ d_obs)
    p_chi2 = chi2.sf(W_obs, df=2)

    # Bootstrap p-value
    d_boot_null = d_boot - mean_boot[None, :]
    W_boot_null = np.einsum("ni,ij,nj->n", d_boot_null, cov_inv, d_boot_null)
    p_boot = (np.sum(W_boot_null >= W_obs) + 1) / (len(W_boot_null) + 1)

    cos_angle = np.clip(np.dot(b_x, b_y), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_angle))

    return {
        "angle_deg": angle_deg,
        "p_chi2": p_chi2,
        "p_boot": p_boot,
        "W_obs": W_obs,
        "d_obs": d_obs,
        "d_boot": d_boot,
        "mean_boot": mean_boot,
        "cov_2d": cov_2d,
    }


# ── Visualization helpers ─────────────────────────────────────────────────────

def vectors_to_az_el(vectors, degrees: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 3D unit vectors to azimuth / elevation angles.

    Flattens a bootstrap cloud on the sphere into two angular coordinates so it
    can be shown as a 2D scatter.

    Parameters
    ----------
    vectors : array-like, shape (n, 3)
        Unit vectors. A single vector must be reshaped to (1, 3) by the caller.
    degrees : bool
        Return degrees rather than radians.

    Returns
    -------
    az, el : numpy.ndarray, shape (n,)
        Azimuth in the xy-plane and elevation above it.
    """
    v = np.asarray(vectors)
    az = np.arctan2(v[:, 1], v[:, 0])
    el = np.arcsin(np.clip(v[:, 2], -1, 1))
    if degrees:
        az, el = np.degrees(az), np.degrees(el)
    return az, el


def cone_half_angle(axis, axis_boot, q: float = 95) -> Tuple[float, np.ndarray]:
    """Half-angle of the confidence cone around a fitted axis.

    The angle containing ``q`` percent of the bootstrap directions — a single
    number for how well-determined the axis direction is. Widens as the
    underlying gradient gets noisier or the sample smaller.

    Parameters
    ----------
    axis : array-like, shape (3,)
        Observed axis direction.
    axis_boot : array-like, shape (n_boot, 3)
        Bootstrap directions, already hemisphere-aligned to ``axis``.
    q : float
        Percentile of the angle distribution to report (95 for a 95% cone).

    Returns
    -------
    half_angle_deg : float
        The ``q``-th percentile angle, in degrees.
    angles : numpy.ndarray, shape (n_boot,)
        Angle of every bootstrap direction from ``axis``, in degrees.
    """
    axis = _unit(axis)
    angles = np.array([
        np.degrees(np.arccos(np.clip(np.dot(_unit(b), axis), -1, 1)))
        for b in axis_boot
    ])
    return np.percentile(angles, q), angles


def plot_projected_arrow_with_cone(
    ax, origin, axis, axis_boot, dims, *,
    color="red", scale: float = 0.8, head_width: float = 0.08,
    head_length: float = 0.12, cone_q: float = 95, cone_alpha: float = 0.18,
    label: Optional[str] = None,
) -> Dict:
    """Plot one axis as an arrow with its confidence cone, projected to a plane.

    The cone is built in 3D around the axis at the bootstrap half-angle, then
    projected — so on an anatomical plane a cone can look wide simply because
    the axis points out of that plane.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    origin : array-like, shape (2,)
        Arrow base in the plane's coordinates (e.g. the LC mesh centroid).
    axis : array-like, shape (3,)
        Axis direction to draw.
    axis_boot : array-like, shape (n_boot, 3)
        Bootstrap directions, used for the cone width.
    dims : sequence of int
        The two coordinate indices defining the plane (e.g. ``[1, 2]`` for
        sagittal, using the ML/AP/DV = 0/1/2 convention).
    color : color
        Arrow and cone color.
    scale : float
        Arrow length in plane units.
    head_width, head_length : float
        Arrowhead geometry.
    cone_q : float
        Percentile for the cone half-angle.
    cone_alpha : float
        Fill opacity of the projected cone.
    label : str, optional
        Legend label for the arrow.

    Returns
    -------
    dict
        ``half_angle_deg``, the cone half-angle actually drawn.
    """
    axis = _unit(axis)
    v2 = axis[list(dims)] * scale

    half_angle_deg, _ = cone_half_angle(axis, axis_boot, q=cone_q)

    # Cone boundary on unit sphere
    e1_3d = np.zeros(3)
    if abs(axis[0]) < 0.9:
        e1_3d[0] = 1.0
    else:
        e1_3d[1] = 1.0
    e1_3d = e1_3d - np.dot(e1_3d, axis) * axis
    e1_3d = _unit(e1_3d)
    e2_3d = _unit(np.cross(axis, e1_3d))

    theta = np.linspace(0, 2 * np.pi, 240)
    alpha_rad = np.radians(half_angle_deg)
    cone_3d = (
        np.cos(alpha_rad) * axis[None, :]
        + np.sin(alpha_rad) * (
            np.cos(theta)[:, None] * e1_3d[None, :]
            + np.sin(theta)[:, None] * e2_3d[None, :]
        )
    )
    cone_2d = cone_3d[:, list(dims)] * scale + origin[None, :]

    poly = np.vstack([origin[None, :], cone_2d, origin[None, :]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=cone_alpha, linewidth=0)
    ax.plot(cone_2d[:, 0], cone_2d[:, 1], color=color, alpha=0.6, linewidth=1)

    ax.arrow(origin[0], origin[1], v2[0], v2[1],
             head_width=head_width, head_length=head_length,
             fc=color, ec=color, linewidth=2, length_includes_head=True, label=label)

    return {"half_angle_deg": half_angle_deg}


# ── Projection scatters ───────────────────────────────────────────────────────

def get_regression_CI(x, y, n_pts: int = 100, ci: float = 0.95):
    """OLS fit line plus a pointwise confidence band for the mean response.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Predictor and response.
    n_pts : int
        Number of points at which to evaluate the fitted line.
    ci : float
        Confidence level for the band.

    Returns
    -------
    y_fit, x_fit, lo, hi : numpy.ndarray
        Fitted values, their x grid, and the lower/upper band.
    """
    from scipy import stats as sp_stats
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    x_fit = np.linspace(x.min(), x.max(), n_pts)
    X_fit = sm.add_constant(x_fit)
    y_fit = res.predict(X_fit)

    y_var = np.array([xi @ res.cov_params() @ xi for xi in X_fit])
    se = np.sqrt(y_var)

    t_crit = sp_stats.t.ppf((1 + ci) / 2, res.df_resid)
    return y_fit, x_fit, y_fit - t_crit * se, y_fit + t_crit * se


def plot_projection_scatter(
    coords, feature_values, feature_name,
    axes_dict_structural,
    colors_structural,
    *,
    center_coords_on_mesh=None,
    figsize_per_panel: Tuple[float, float] = (5, 5),
):
    """Project units onto each structural axis and scatter against a feature.

    One panel per structural axis: each location's scalar position along that
    axis (x) against its feature value (y), with a Spearman correlation and an
    OLS fit + CI band. Answers whether units sitting at different points along
    a structural gradient differ in the feature.

    Parameters
    ----------
    coords : array-like, shape (n, 3)
        Bregma-centered LPS mm coordinates, one row per unit.
    feature_values : array-like, shape (n,)
        Scalar feature per unit (e.g. ``T_rt``).
    feature_name : str
        Used for the y-axis label and panel titles.
    axes_dict_structural : dict
        Maps axis name to its (3,) unit vector. One panel per entry.
    colors_structural : dict
        Maps axis name to a color, used for the x-axis label and ticks.
    center_coords_on_mesh : array-like, shape (3,), optional
        Subtracted from ``coords`` before projecting (e.g. the LC mesh
        centroid), so projections are centered on the structure rather than
        on bregma.
    figsize_per_panel : tuple of float
        Width and height of each panel, in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    results : dict
        Per axis name, ``r`` (Spearman), ``p`` and ``n``.
    """
    n_axes = len(axes_dict_structural)
    fig, axes_arr = plt.subplots(1, n_axes, figsize=(figsize_per_panel[0] * n_axes, figsize_per_panel[1]))
    if n_axes == 1:
        axes_arr = [axes_arr]

    valid_all = np.isfinite(feature_values) & np.all(np.isfinite(coords), axis=1)

    results = {}

    for ax, (axis_name, axis_vec) in zip(axes_arr, axes_dict_structural.items()):
        color = colors_structural.get(axis_name, "gray")

        # Project coordinates onto this axis
        c = np.asarray(coords, dtype=float).copy()
        if center_coords_on_mesh is not None:
            c = c - center_coords_on_mesh

        proj = c @ axis_vec  # scalar projection per unit

        valid = valid_all & np.isfinite(proj)
        proj_v = proj[valid]
        feat_v = np.asarray(feature_values, dtype=float)[valid]

        # Color points by feature value (diverging if signed)
        has_neg = np.any(feat_v < 0)
        if has_neg:
            amp = np.nanquantile(np.abs(feat_v), 0.95)
            if amp == 0:
                amp = 1.0
            norm = plt.Normalize(vmin=-amp, vmax=amp)
            cmap = "coolwarm"
        else:
            norm = None
            cmap = "viridis"

        ax.scatter(
            proj_v, feat_v,
            c=feat_v, cmap=cmap, norm=norm,
            s=25, alpha=0.7, edgecolor="white", linewidth=0.3,
        )

        # Regression line + CI
        if len(proj_v) > 4:
            r, p = spearmanr(proj_v, feat_v)
            y_fit, x_fit, ci_lo, ci_hi = get_regression_CI(proj_v, feat_v)
            ax.plot(x_fit, y_fit, color="black", linewidth=1.5)
            ax.fill_between(x_fit, ci_lo, ci_hi, color="black", alpha=0.15)

            title_color = "red" if p < 0.05 else "black"
            ax.set_title(
                f"{feature_name} vs {axis_name}\n"
                f"r={r:.2f}, p={p:.3g}",
                color=title_color, fontsize=11,
            )
            results[axis_name] = {"r": r, "p": p, "n": len(proj_v)}
        else:
            ax.set_title(f"{feature_name} vs {axis_name}\nInsufficient data")
            results[axis_name] = {"r": np.nan, "p": np.nan, "n": len(proj_v)}

        ax.axhline(0, ls=":", color="gray", lw=0.5)
        ax.axvline(0, ls=":", color="gray", lw=0.5)

        ax.set_xlabel(f"Projection on {axis_name}", color=color, fontsize=10)
        ax.set_ylabel(feature_name)
        ax.tick_params(axis="x", colors=color)

    plt.tight_layout()
    return fig, results
