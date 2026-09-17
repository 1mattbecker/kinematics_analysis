# fip_* to-do

Deferred work scoped to the `fip_*` notebook series (`fip_00_explore.ipynb`,
`fip_01_movement_value_coding.ipynb`, `fip_02_ne_only_events.ipynb`). Kept separate from
`TODO.md`/`REORG.md`, which track the `kin_*`/`eph_*` port plan. Newest first, dated
`YYYY-MM-DD`.

---

## 2026-09-15 (1)

**The `fip_utils.py` extraction logged below is DONE** (see `CHANGELOG.md` 2026-09-15 (3)).
`code/fip_utils.py` now carries the shared setup and all three notebooks import it. Remaining
`fip_*` work, newest first:

- **Pin `environment/Dockerfile` to a rachel-analysis-utils commit that still has the JSON
  curation API.** NOT DONE — CLAUDE.md puts `/environment` off limits, so this needs a human.
  Upstream `main` (`b7b7487`, 2026-09-14, "CSV-based data curation inputs") deleted
  `apply_curation_nwb_list`; `data_curation_helpers.py` is now `load_curation` +
  `apply_curation_df_fip`. The Dockerfile installs `@main` unpinned, and the cached image still
  has the old API — the next rebuild breaks `fu.load_curated_sessions`, and then
  `parse_event` / `build_meta` / `pick_example`, which key off `event` looking like
  `G_1_dff-poly` and off `intended_measurement` existing. Change line 60 to:

      -e git+https://github.com/AllenNeuralDynamics/rachel-analysis-utils.git@864550d55356ecb4906d05801cdae694bd000fde#egg=rachel-analysis-utils

  Pin `864550d` (2026-09-02) verified to carry: `apply_curation_nwb_list(nwb_list, curation,
  drop_borderline=False)` with the signature we call; all five private helpers in `nwb_utils`
  (so `fu.patch_curation_helpers` resolves); `DA_NE_4channel_datacuration_firstpass.json`; and
  `analysis_utils.py` at 177 lines, 3.9-compile-clean, with `enrich_df_trials`. The adjacent
  Dockerfile comment ("no deps", hence the isolated `--ignore-requires-python` layer) stays
  accurate: `dependencies = []` at the pin *and* at main. Note the failure mode at main is not a
  resolution conflict but an undeclared import — `data_curation_helpers.py` does
  `from aind_bwnm_fiber_data_curation_utils import data` at module top level while declaring it
  nowhere, so pip installs nothing for it and the import raises `ModuleNotFoundError`.

- **Migrate curation to the CSV API and unpin.** A real migration, not a swap: the new
  `apply_curation_df_fip` overwrites `df_fip['event']` with target names, needs a `patch_cord`
  column from `nwb_utils.split_fiber`, wants to run *before* `enrich_fip_in_df_trials`, and
  drops `intended_measurement` — so `fu.parse_event`, `fu.build_meta` and `fu.pick_example` all
  need rework, and the new dependency must be added to the image.

- **Pass 2: move `fip_00`'s multi-session section onto the upstream PSTH machinery.**
  `session_etr_mean` + `aggregate_series` + `collect_region_etr` + `plot_by_subject` (the four
  functions deliberately left in `fip_00` cell 36 rather than moved into `fip_utils`) collapse
  into `plot_fip.plot_fip_psth_compare_alignments(nwb_list, alignments=[per-session dicts],
  channel=..., data_column="data_z", error_type="sem_over_sessions")`. Compatible: Rachel's
  `get_dummy_nwbs` builds `dummy_nwb(df_trials_i, df_events_i, df_fip_i)`, so `df_events` exists
  and the `nu.create_df_events` branch never fires. Requires ME as a pseudo-channel for every
  session, i.e. flipping on `BUILD_NWB_LIST_ME` (already implemented in `fip_00`'s last cell);
  its memory cost must be re-measured. Use `error_type="sem_over_sessions"` first — upstream's
  own docstring advises deferring hierarchical bootstrap until analyses are finalized — with
  `hb_sem` + `aggregate_bootstrap_statistics` (per-timepoint p-values, which we have no
  equivalent for today) held for final figures. **Expect figures to change:**
  `fip_psth_inner_compute` hardcodes `output_sampling_rate=40` vs. `fu.peri_event`'s `fs=20`,
  and the error band changes from a manual session->subject rollup. That is why it is a separate
  pass from the extraction.
  `window_mean` and Analysis 3 stay hand-rolled — `trial_metrics.get_average_signal_window` adds
  a per-trial column to `df_trials`, whereas `window_mean` averages an ETR-mean Series over
  relative time. Different semantics, not a substitute.

- **Run Pass 1 on Code Ocean.** The extraction is verified statically only (py_compile,
  nbformat, nbconvert+compile, pyflakes); nothing has executed against real data. Run `fip_02`,
  then `fip_01`, then `fip_00` *including* the multi-session section, and confirm:
  `process_session` still skips ME-less sessions gracefully; Analyses 1-4 and the cross-session
  table reproduce; `fip_02`'s NE-only / NE+DA counts match their pre-refactor values (the
  `ddof=0`->`ddof=1` change should move no onsets — if counts shift by more than a couple of
  events, investigate rather than accept); and peak RSS after `fu.load_curated_sessions()` stays
  at or below the ~43 GB measured with the old in-notebook `del`.

- **"Anticipatory movement during CS+Delay"** — the 4th figure originally requested alongside
  `fip_01`'s three, still skipped. This dynamic-foraging task has no literal CS+/delay epoch;
  the closest structural analog is the goCue->choice RT window. Revisit once there is a clearer
  definition of what epoch/comparison is wanted.

## 2026-09-14 (2) — superseded, kept for context


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

## 2026-09-14 (1) — superseded, kept for context

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
