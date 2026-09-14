# fip_* to-do

Deferred work scoped to the `fip_*` notebook series (`fip_00_explore.ipynb`,
`fip_01_movement_value_coding.ipynb`). Kept separate from `TODO.md`/`REORG.md`, which track the
`kin_*`/`eph_*` port plan. Newest first, dated `YYYY-MM-DD`.

---

## 2026-09-14

- **Extract shared setup into `code/fip_utils.py`.** `fip_00_explore.ipynb` and
  `fip_01_movement_value_coding.ipynb` currently duplicate ~150 lines of setup (session load,
  curation + the `apply_curation_nwb_list` missing-helper patch, `motion_energy_to_session`,
  `threshold_onsets`, `peri_event`, `zscore`, `attach_me_to_df_fip`). Factor into a flat module
  once `fip_01`'s approach has stabilized, following the `plotstyle.py`/`encoding_plots.py`
  pattern already used in this repo. Note: `fip_01`'s `attach_me_to_df_fip` stores raw `me`
  (not `me_z` like `fip_00`'s) — reconcile which convention the shared version uses when this
  is extracted (see `fip_01`'s Deviations-from-Rachel's-pipeline note in its implementation
  plan for why raw is preferred).
- **Verify `import rachel_analysis_utils.analysis_utils` on Code Ocean.** A local
  `python3.9 -m py_compile` against a fresh clone of `main` succeeds cleanly — the nested-quote
  f-string bug `TODO.md`'s Python-3.11-upgrade section cites (`analysis_utils.py:294`) appears
  to have been fixed upstream (current file is 177 lines). `fip_01_movement_value_coding.ipynb`
  already tries `enrich_df_trials` directly with a local fallback; if the CO run confirms the
  import works, (a) switch `fip_00_explore.ipynb`'s multi-session section from its local
  `enrich_streaks` reimplementation to `enrich_df_trials` directly, and (b) update `TODO.md`'s
  Python-3.11-upgrade rationale — item 2 no longer blocks on this specific function, though the
  base-image upgrade may still be worth doing for other reasons.
- **"Anticipatory movement during CS+Delay"** — the 4th figure originally requested alongside
  `fip_01`'s three, skipped for that first pass. This dynamic-foraging task has no literal
  CS+/delay epoch; the closest structural analog is the goCue→choice RT window (movement
  building in anticipation of choice/outcome, analogous to anticipatory licking in Pavlovian
  paradigms). Revisit once there's a clearer definition of what epoch/comparison is wanted.
