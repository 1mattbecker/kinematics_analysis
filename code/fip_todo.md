# fip_* to-do

Deferred work scoped to the `fip_*` notebook series (`fip_00_explore.ipynb`,
`fip_01_movement_value_coding.ipynb`, `fip_02_ne_only_events.ipynb`,
`fip_03_da_ne_commonality.ipynb`, `fip_04_da_ne_xcorr.ipynb`). Kept separate from `TODO.md`/`REORG.md`, which track the
`kin_*`/`eph_*` port plan. Newest first, dated `YYYY-MM-DD`.

---

## 2026-09-25

**`fip_04_da_ne_xcorr.ipynb` added** (see `CHANGELOG.md` 2026-09-25). Synthetic-validated only.

- **Run it on Code Ocean** and check the side table in §2: expect roughly 7 animals (813929 has
  both PL fibers dropped, 809488 is `drop_all`). Confirm which side each animal lands on matches
  the curation notes; `HEMI_BY_SUBJECT` overrides.
- **Check the curation drop semantics.** `drop_channels` lists physical channels (`G_0`, `G_1`),
  and for 808054/808056 most sessions carry `misconnect_fixes` that move PL to a different
  physical channel. If a drop is applied before the remap, the surviving PL fiber in those
  sessions may be the noisy one. §2's side table shows what survived; compare with the notes.
- **Contralateral control.** Every animal also has the opposite-side NAc dLight. DA(contra) × NE
  would show whether the coupling is lateralized. Bilateral dLight is r ≈ 0.98 in 808054, so a
  near-identical result is the expected outcome.
- The Wilcoxon on peak lag has many exact-zero ties, so scipy falls back to the normal
  approximation and warns. Report the lag with the bootstrap CI if the test is unstable.

---

## 2026-09-23

From the first real Code Ocean run of `fip_03` (see `CHANGELOG.md` 2026-09-23):

- **Ask Rachel two things.** What window length the `pearsonR` series uses — its DA x NE value
  is +0.034 against section 3's +0.29, and a rolling window shorter than the ~2-8 s shared
  component would explain the gap entirely. And what `bright` denotes in the variant name
  `dff-bright_mc-iso-IRLS` (the rest reads as dF/F, motion-corrected against the isosbestic
  by iteratively reweighted least squares).
- **`NAc(L)-dLight` x `NAc(R)-dLight` = +0.983** across all 10 sessions, with `max|diff|` = 0.77
  so they are not duplicates. Bilateral NAc DA at r = 0.98 is high enough to be worth raising;
  subject 808054 is the one with per-session `misconnect_fixes` in curation.
- **DA is recorded bilaterally and `pick_example` picks a hemisphere by tie-break** — both
  dLight channels have identical sample counts, so it takes whichever sorts first. The two are
  ~98% identical so section 5/6 results barely depend on it, but the choice should be explicit.
- **Should sections 5 and 6 be high-passed?** Motion energy has its own slow structure
  (engagement declining across a session). If DA and NE each track it independently, drift
  inflates the *unique* components rather than the shared one. Run section 6 both ways and
  compare; if the components move materially that difference belongs in section 7.
- **One session has a channel labelled `no_fiber`**, and `NAc(L)-rAch` appears in 9 of 10
  sessions. Curation gaps, not urgent.

---

## 2026-09-22

**`fip_03_da_ne_commonality.ipynb` added** (see `CHANGELOG.md` 2026-09-22). Validated statically
and against synthetic data; never run on Code Ocean. Open items:

- **Run it on Code Ocean.** Run `fip_02` first as a smoke test — no `fip_*` notebook has executed
  against real data (item 4 below), so the first real run of `fip_03` will surface any
  `fip_utils` bug its predecessors would have found. Expect ~15 min after
  `load_curated_sessions()`. Confirm §2 prints 10 sessions / 1 subject.
- **Sweep the two switches.** Re-run with `ME_TRANSFORM = "raw"` and compare §6's partition
  against the `log1p` run; if the three components move materially, motion energy's right tail is
  driving the split and §7 should say so. Then re-run with `TARGET = "events"` and confirm the
  Poisson path reports D² in place of R² throughout.
- **Animal-level inference.** The binding constraint is that motion energy exists for 10 sessions
  from one animal. Two routes: swap the behavioral target to lick/choice events from `df_trials`
  (~172 curated sessions, 20 subjects), which needs a different event source in §2 and a
  bootstrap over animals in §6; or run `aind-motion-energy` (capsule 5110154) over more of the
  20 subjects' behavior-video assets to widen the existing path.
- **Figure destination.** §1 writes to `/root/capsule/scratch/figures/fip`, which a reproducible
  run does not preserve. Anything headed for a paper figure needs copying to `/results`.
- **Open parameter choices**, all with defaults in place and none yet checked against real data:
  the lag basis (`N_BASIS = 8`, log-warped over −2 to +5 s), block CV geometry (`N_BLOCKS = 5`,
  `EMBARGO_S = 7.0`), and whether `fu.pick_example` silently taking one fiber per region is
  acceptable when a session has two dLight fibers.

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
