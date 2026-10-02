# fip_* / men_* to-do

Open work for the FIP and motion-energy notebooks (`fip_00`–`fip_07`, `men_00`). `TODO.md` tracks
the `kin_*` / `eph_*` work. History is in `CHANGELOG.md`. Last revised 2026-10-02.

---

## Plan: motion energy × FIP

All analyses run on one dataset: the CSV-curated assets (`fu.load_pairs`, 9 animals, 97 sessions)
with the aligned motion-energy table (`fu.motion_energy_to_session`). The animal is the unit.
Steps 2–5 share one new notebook (`fip_08`).

1. **Check the motion-energy data.** Attach the 4-channel rebuild as `DANE_4channels_curated`,
   then run `men_00` on Code Ocean. Use its lick-triggered check to confirm the clock in every
   session and camera, and pick the camera. Decide on 808054_2025-09-05 (odd ME spectrum, few
   onsets) and animal 816212 (outlier in every coupling measure; refused cameras).
2. **FIP variance explained by movement beyond the task.** Add lagged motion energy to the task
   model in `fu.task_residuals` (go cue, outcome, lick kernels) and measure the variance it adds,
   for DA and for NE, per animal. Compare with and without slow drift removed, so engagement does
   not inflate it. This replaces porting `fip_03`.
3. **Movement as a confound of the RPE results.** Add motion energy in the outcome window to
   `fu.fit_rpe_terms` and check whether NE's RPE slope and DA's reward term change.
4. **NE solo transients.** Align motion energy to solo and partnered NE and DA transients
   (`fip_05` section 5), across animals and by task context.
5. **Movement with and without licking.** Align DA and NE to motion-energy events without
   licking, and to instructed vs uninstructed lick bouts (`men_00` definitions).
6. **Timing around movement onset.** DA and NE relative to motion-energy onsets, including onsets
   in the go cue → choice window ("anticipatory movement"). Lags describe the measured signals;
   GCaMP and dLight kinetics differ.
7. **Slow coupling.** Running-mean motion energy as the engagement covariate for tonic DA/NE and
   for `fip_05`'s negative 2–10 min residual correlation.

Not planned: porting `fip_03` as it is (step 2 supersedes it), more single-session work in
`fip_00`–`fip_02`, directional or causal claims.

---

## Open items

### Code Ocean runs
- Run `fip_00`–`fip_04` with the 2026-10-02 helper modules; they have only been lint-checked.
  `fip_03` rebuilds its cache (file renamed).
- Copy figures headed for a paper from `scratch/figures` to `/results`.

### Questions for Rachel
- 808056 is missing from the 4-channel rebuild; it was the most strongly coupled animal in `fip_04`.
- Is `latNAcc` the same placement as the older asset's `NAc`?
- The RPE-split finding (`fip_06`): RPE = 0 omissions in her RPE ≥ 0 group, and 37/301 sessions
  with `forget_rate_unchosen` fit at 1.0.
- What window the `pearsonR` series uses, and what `bright` means in `dff-bright_mc-iso-IRLS`.

### Analyses
- Move `fip_05` onto Rachel's code (agreed): `dummy_nwb.load`, her `data_z` / `data_z_norm`,
  `event_triggered_response`, her RPE slope with the outcome split and pre-first-reward trials
  dropped; keep in `fip_utils` only what the libraries don't have.
- Contralateral control: DA from the opposite NAc × NE.
- Behavior-model dependence: RPE and value come from one Q-learning fit; a refit or another model
  family would change `fip_05` sections 1–2.
- Indicator kinetics: deconvolve with published kernels, or compare with an nLight/GRAB-NE cohort.
- Decide whether `fip_04` (older asset) is retired in favour of `fip_07`.

### Older asset (`DA_NE_4channels`, JSON curation)
- Bilateral NAc dLight correlates at r ≈ 0.98 in 808054, and `pick_example` picks a hemisphere by
  tie-break; make the choice explicit if this asset stays in use.
- One session has a `no_fiber` channel; `NAc(L)-rAch` is missing from one of 808054's sessions.
- Moving to rachel-analysis-utils' CSV curation API would need `parse_event`, `build_meta` and
  `pick_example` reworked. Only needed if this asset stays in use; the environment is pinned to
  `864550d`, which still has the JSON API.

### Library
- Move the bout helpers in `behavior_utils` into the video-analysis library (PR, then rebuild).

---

## Closed

- `fip_utils` extraction; `fip_coupling` merged into `fip_utils`; shared helper modules
  (2026-10-02, `helper_cleanup.md`).
- Dockerfile pinned to rachel-analysis-utils `864550d`.
- Rachel's `analysis_utils` imports on Python 3.12; local fallback removed.
- "Pass 2" onto `plot_fip`: not done, by decision. It pools over sessions and needs motion energy as
  a `df_fip` channel; `fip_00` uses `stats_utils` / `plot_utils`.
- `fip_04`'s 5–10 Hz coherence: noise floor of low-passed signals (`fip_05`).
- `fip_05`'s "NE 0.2 s after DA": produced by DA's omission dip; latencies match on rewarded trials.
- `fip_03` with motion energy for one animal only: superseded by the motion-energy table and plan
  step 2. Its open choices (switch sweeps, lag basis, CV geometry, high-pass) carry into step 2.
