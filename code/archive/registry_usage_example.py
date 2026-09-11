"""
Registry usage example — shows how existing notebook code maps to
PerUnitStatsRegistry calls.

This is NOT a standalone script. It's a reference for refactoring the
notebook. Copy the relevant sections into your notebook cells.
"""

# =============================================================
# 1. SETUP (one cell, after your existing imports + data loading)
# =============================================================

from per_unit_stats_registry import PerUnitStatsRegistry

reg = PerUnitStatsRegistry(
    get_session_prefix=get_session_prefix,
    alpha=0.05,
)


# =============================================================
# 2. REGISTER YOUR CORRELATION RESULTS
#    (replaces manual handling of cor_df objects)
# =============================================================

# These calls happen right after each run_and_plot_for_predictor call.
# The run_and_plot_for_predictor output dict is passed directly.

reg.register_cor_df("spkct_vs_rt", spkct_rt)
reg.register_cor_df("spkct_vs_rt_bl", spkct_rt_bl)
reg.register_cor_df("spklat_vs_rt", spklat_rt)
reg.register_cor_df("spkct_bl_vs_bl_rate", spkct_bl)

# Multi-predictor sweep results (from the pairs dict loop)
for key, y_col in pairs.items():
    res = run_and_plot_for_predictor(
        all_counts_df,
        x_col="spike_count",
        y_col=y_col,
        example_n=0,
        alpha=0.05,
        filter_query="reaction_time_firstmove < 2",
        summary_metric='spearman_t',
    )
    reg.register_cor_df(f"spkct_vs_{key}", res)


# =============================================================
# 3. REGISTER PARTIAL CORRELATIONS
# =============================================================

pc_vel_pRT = analyze_unit_partial_correlations(
    df_filtered,
    x_col="spike_count",
    y_col="first_move_mean_velocity",
    control_col="reaction_time_firstmove",
    method="spearman",
    min_trials=50,
)
reg.register_partial("velocity_partial_rt", pc_vel_pRT)


# =============================================================
# 4. REGISTER SUE'S TABLE (bulk)
#    (replaces cells 83-86 merge + cells 88-89 screening setup)
# =============================================================

# One call registers ALL T_ columns from Sue's table.
# Each becomes "sue::{suffix}", e.g. "sue::baseline_hit_all"
n_registered = reg.register_sue(sue_plus)
print(f"Registered {n_registered} entries from Sue's table")

# See what we have:
print(reg)


# =============================================================
# 5. REGISTER YOUR OLS RT REGRESSIONS
#    (the T_rt / T_rt_bl columns from build_sue_plus_with_rt_models)
# =============================================================

# These are already in sue_plus after build_sue_plus_with_rt_models.
# Register them as separate entries so they're on equal footing.
reg.register_regression(
    "ols_rt_response",
    sue_plus,
    t_col="T_rt",
    p_col="p_rt",
    coef_col="coef_rt",
    q_col="q_rt",
    n_col="n_trials",
)
reg.register_regression(
    "ols_rt_baseline",
    sue_plus,
    t_col="T_rt_bl",
    p_col="p_rt_bl",
    coef_col="coef_rt_bl",
    q_col="q_rt_bl",
    n_col="n_trials_bl",
)


# =============================================================
# 6. COMPARISONS — replaces scattered comparison code
# =============================================================

# --- BEFORE (cell 47): ---
# merged, fig = scatter_with_marginals_cor_compare(
#     spkct_rt['cor_df'], spklat_rt['cor_df'],
#     alpha=0.05,
#     label_a="spike count vs RT",
#     label_b="first-spike latency vs RT",
#     use_spearman=True,
#     metric="t",
#     bins=20,
#     title="LC Population Correlation: Reaction Time\n(t-statistics)",
# )

# --- AFTER: ---
merged = reg.compare("spkct_vs_rt", "spklat_vs_rt",
                      title="LC Population: spike count vs latency RT correlations")


# --- BEFORE (cell 53): ---
# cor_df_count = spkct_rt["cor_df"]
# cor_df_count_bl = spkct_rt_bl["cor_df"]
# merged, fig = scatter_with_marginals_cor_compare(
#     cor_df_count, cor_df_count_bl, ...)

# --- AFTER: ---
merged = reg.compare("spkct_vs_rt", "spkct_vs_rt_bl",
                      title="Response vs baseline spike count ~ RT")


# --- BEFORE (cell 52): ---
# merged = compare_unit_correlations(
#     cor_df_count, cor_df_lat,
#     label_a="rate-RT", label_b="first-spike-RT", ...)

# --- AFTER (same functionality, cleaner): ---
merged = reg.compare("spkct_vs_rt", "spklat_vs_rt")


# --- Cross-source comparison (your Spearman vs Sue's encoding): ---
# BEFORE: required manual merge on session_prefix + unit,
#         column renaming, ad hoc scatter_with_marginals calls

# AFTER:
merged = reg.compare("spkct_vs_rt", "sue::baseline_hit_all")
merged = reg.compare("ols_rt_response", "sue::outcome_com_ori")


# =============================================================
# 7. SCREENING — replaces spearman_screen_T_cols / spearman_screen_T_vs_T
# =============================================================

# --- BEFORE (cell 89): ---
# my_cols = ["rt__spearman_t", "rt_bl__spearman_t"]
# def spearman_screen_T_cols(df, my_cols, t_prefix="T_", ...): ...
# top5_dict = spearman_screen_T_cols(sue_plus, my_cols, ...)

# --- AFTER: ---

# Screen your Spearman RT t-stats against all of Sue's entries
screen_rt = reg.screen("spkct_vs_rt", source="sue", top_n=10)
print(screen_rt)

# Screen against all entries (including your own)
screen_all = reg.screen("spkct_vs_rt", top_n=15)

# Screen with a name prefix filter
screen_baseline = reg.screen("spkct_vs_rt", prefix="sue::baseline")


# --- BEFORE (cell 98): ---
# my_cols = ["T_rt", "T_rt_bl"]
# def spearman_screen_T_vs_T(df, my_cols, t_prefix="T_", ...): ...

# --- AFTER: ---
screen_ols = reg.screen("ols_rt_response", source="sue", top_n=10)
screen_ols_bl = reg.screen("ols_rt_baseline", source="sue", top_n=10)


# =============================================================
# 8. SUMMARIES
# =============================================================

# Quick summary of any entry
reg.summary("spkct_vs_rt")
reg.summary("sue::baseline_hit_all")

# List everything
print(reg)
