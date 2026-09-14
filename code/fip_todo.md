# fip_* to-do

Deferred work scoped to the `fip_*` notebook series (`fip_00_explore.ipynb`,
`fip_01_movement_value_coding.ipynb`, `fip_02_ne_only_events.ipynb`). Kept separate from
`TODO.md`/`REORG.md`, which track the `kin_*`/`eph_*` port plan. Newest first, dated
`YYYY-MM-DD`.

---

## 2026-09-14 (2)

- **`fip_utils.py` extraction is now overdue.** A 3rd notebook (`fip_02_ne_only_events.ipynb`)
  duplicates the same setup cells (data loading, curation + memory fix, session select,
  `pick_example`/`EXAMPLE_SPECS`, motion-energy-on-the-FIP-clock, `zscore`/`threshold_onsets`/
  `peri_event`) already duplicated once between `fip_00` and `fip_01`. Do the extraction logged
  below before a 4th notebook makes it worse.
- `fip_02_ne_only_events.ipynb` deliberately does **not** run the `enrich_fip_in_df_trials`/
  `remove_tonic_df_fip` per-trial baseline pipeline `fip_01` uses — it only needs continuous-time
  `data_z` (from `zscore_fip` directly), since its analysis is onset-based, not per-trial. Worth
  keeping in mind when `fip_utils.py` is extracted: the shared module should expose the
  z-scoring/attach step and the per-trial pipeline as separable pieces, not one bundled function.

## 2026-09-14 (1)

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
