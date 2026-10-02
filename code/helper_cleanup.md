# Helper clean-up checklist (started 2026-10-02)

From the duplicated-function audit of 2026-10-02. Goal: one implementation per operation, in the
module where it belongs, using the AIND libraries first. Results are preliminary, so outputs may
change; each item is checked for working as intended, not for matching earlier numbers.

## Target layout

| Module | Owns |
|---|---|
| `signal_utils.py` (new) | Generic time-series helpers, numpy/scipy only: z-score, threshold onsets, binning onto a grid, grid peri-event / window means, rolling mean, cross-correlation (linear and circular, one lag convention), contiguous runs, event coincidence and its circular-shift null, transient detection |
| `stats_utils.py` (new) | Per-animal averaging (session → animal → grand mean ± SEM), Wilcoxon over animals, hierarchical bootstrap (wrapping `aind_hierarchical_bootstrap`), FDR (statsmodels), Wilson CI, p-value formatting |
| `plot_utils.py` (new) | Summary plots: mean ± SEM band over animals, strip plots by measure and by animal (`plotstyle.py` stays style-only, as its docstring says) |
| `behavior_utils.py` (new) | Task-behaviour helpers used by several series: lick bouts (library), bout classification, task context of event times |
| `fip_utils.py` | All FIP code: absorbs `fip_coupling.py` (which is deleted) |
| `men_utils.py` | Motion-energy series only (inventory, loading, normalisation); generic pieces move out |
| `kin_utils.py` (new) | `kin_*` helpers: `coerce_bool`, `load_kps_raw`, `jaw_position` (numpy/pandas only, so `kin_*` keep running locally; `data_loading` imports the video library) |

## Checklist

### A. Module layout
- [x] A1. Create `signal_utils.py`; move the generic helpers from `fip_utils`, `fip_coupling`, `men_utils`
- [x] A2. Create `stats_utils.py`; move the per-animal / test helpers
- [x] A3. Merge `fip_coupling.py` into `fip_utils.py`; switch `fip_05`, `fip_07`, `men_00` to `fu`
- [x] A4. Plot helpers (`plot_mean_sem`, `grand_trace`, `animal_strip` ×3, `session_strip`) into `plot_utils.py`
- [x] A5. Create `behavior_utils.py`: lick bouts, `classify_bout_times` / `annotate_movement_bouts` (from `ephys_utils`), `label_context`

### B. Use the libraries
- [x] B1. `men_00` lick bouts from basic-analysis `annotate_lick_bouts` (0.7 s) and labels from `ephys_utils.classify_bout_times`; drop `men_utils.lick_bouts` / `label_bouts`
- [x] B2. `fip_04.hier_boot_mean` → `aind_hierarchical_bootstrap`
- [x] B3. `test_session_quality.get_session_prefix` (×3) → library `tongue_ephys.get_session_prefix`
- [x] B4. `kin_00.resolve_labeled_video` → library `find_labeled_video`
- [x] B5. `fu._enrich_trials_fallback` → Rachel's `enrich_df_trials` only
- [x] B6. `_fdr_bh` (×2, `encoding_methods`, `per_unit_stats_registry`) → statsmodels `multipletests`
- [ ] B7. `fip_00` multi-session ETR helpers → `plot_fip` PSTH machinery, or record why not

### C. One copy of each helper
- [x] C1. Peri-event / window mean (`fip_coupling`, `men_utils`); NaN-safe window mean
- [x] C2. Cross-correlation: `fu.norm_xcorr` + `fc.residual_xcorr` (opposite lag signs) + 3 circular copies → one convention
- [x] C3. Coincidence + circular-shift null: `fip_02` inline (no minimum shift), `fc.shift_coincidence_null`, `mu.near_any` / `shifted_fraction`
- [x] C4. Z-score: `fc.zscore` (ddof 0, not NaN-safe) vs `fu.zscore`
- [x] C5. Per-animal averaging: `fc.animal_means`, `fip_04.animal_means` / `grand_mean`, `fip_05.stack_animal`, `fip_07.animal_stack`, `mu.animal_traces`; SEM with ddof=1 and per-point n
- [x] C6. `contiguous_runs` + cluster test (`fip_03`, `fip_04`)
- [x] C7. `stars` / `sig_stars` / `fmt_p`; `sess_corr` (`fip_05`, `fip_07`)
- [x] C8. `coerce_bool` (×4 `kin_*`), `load_kps_raw` (`kin_06`, `kin_07`), and `kin_07.get_jaw_y` = `kin_06.load_jaw_from_keypoints` → `kin_utils`
- [ ] C9. `load_example_session_and_unit` (`eph_00`, `eph_07`); `eph_07.count_spikes` vs `ephys_utils.count_spikes_in_window`
- [x] C10. `_canon_unit` (`encoding_methods`, `per_unit_stats_registry`)
- [x] C11. `men_00.event_context` vs `fu.label_context`; `per_animal_timecourse` vs `kin_04._interp_to_grid`
- [ ] C12. `val_03.wilson_ci` → shared; `lickometer_qc.refractory_mask` vs library `filter_timestamps_refractory`
- [ ] C13. Scripts: `list_keys` / `read_json` (`build_me_table`, `build_me_asset_map`), `check_leading_lost_frames.fetch` vs `build_me_table.download`
- [x] C14. Same-notebook redefinitions: `test_session_quality` (`plot_combined_summary_compare` ×3, `show_session_video_reel` ×2), `attach_data._parse_vp_dt` ×2

### D. Finish
- [ ] D1. Run what runs locally (`fip_05`, `fip_06`, `fip_07`, `men_00` on a test table, `kin_*` on `data/for_local`); static checks on the rest
- [ ] D2. CLAUDE.md module table, CHANGELOG, `fip_todo.md`

## Notes as items close

- **A1–A5, C1–C6 (2026-10-02).** `fip_coupling.py` is gone; `fip_05`, `fip_07`, `men_00` import
  `fip_utils` / `signal_utils` / `stats_utils` / `behavior_utils` / `plot_utils`. Also replaced
  along the way: the cross-correlation and coherence shift-null loops that `fip_03`, `fip_04` and
  `fip_07` each ran inline (`su.xcorr_shift_null`, `su.coherence_shift_null`); `fip_07`'s
  `shift_p` / `shift_z`; about a dozen inline mean ± SEM blocks (`pu.plot_mean_sem`, `st.mean_sem`;
  several used ddof=0). One lag convention: `su.norm_xcorr(a, b)` peaks at + lag when `a` is later.
  `fip_03`'s xcorr null now uses a 60 s minimum shift like `fip_04` / `fip_07` (was 1 s).
  `su.window_mean_grid` skips NaN samples (`fip_coupling.window_mean` returned NaN).
  Checked: `fip_05`, `fip_07` run locally, outputs equal to the pre-change run except draws of the
  random shift nulls; `men_00` runs on a synthetic ME table built on 6 real sessions (planted
  lick-locked ME at −50 ms recovered at −0.05 s); `fip_00`–`fip_04` lint clean (no undefined
  names); every moved helper tested against its old copy on synthetic data.
- **B1.** `men_00` bouts now use the library's 0.7 s gap (was 0.5 s) and `classify_bout_times`;
  `men_utils.load_session` keeps each session's lick events; cache version bumped.
- **B2.** `aind_hierarchical_bootstrap` pools the resampled sessions, so its CI is on the
  session-weighted mean; `fip_04`'s table labels it that way. Its own version resampled to the mean
  of animal means.
- **C11.** `men_00`'s `event_context` replaced by `bu.label_context` (the FIP transient contexts).
  `men_00.per_animal_timecourse` (minutes from first cue, per animal) and
  `kin_04._interp_to_grid` (fraction of session, per session) answer different questions; both stay.
- **B3, C14.** `test_session_quality_analysis`: one `plot_combined_summary_compare` and one
  `show_session_video_reel` (the last version of each, in the first cell that calls it; the
  superseded cells are deleted), `get_session_prefix` from the library. `attach_data`: cell 4 uses
  cell 3's helpers instead of redefining them.
- **B4.** `kin_00.resolve_labeled_video` wraps the library's `find_labeled_video`; its JSON lookup
  ended in the same `/root/capsule/data` glob.
- **B5.** Rachel's `enrich_df_trials` imports on 3.12 and gives the same `num_reward_past` as the
  local copy (checked on a real session); the copy is gone.
- **B6, C10.** `encoding_methods` and `per_unit_stats_registry` use `st.fdr_bh`; the registry takes
  `_canon_unit` and the session-prefix function from `encoding_methods` (library, with a local copy
  only when the library is missing). Registry q-values checked against statsmodels.
- **C7, C8.** `fmt_p`, `stars` in `stats_utils` (`eph_09`'s "ns" is now "n.s."). `kin_02`, `kin_05`,
  `kin_06` run locally (`kin_02`/`kin_05` outputs identical to the saved ones); `kin_07` lint only.
- **Follow-up, not in this list:** the bout helpers in `behavior_utils` are library candidates
  (a PR to the video-analysis library, then a capsule rebuild). `attach_data`'s asset search could
  use `aind_dynamic_foraging_data_utils.code_ocean_utils` (`get_assets`, `attach_data`); not changed,
  since it attaches assets and cannot be tested here.
