# fip_* / men_* to-do

Open work for the FIP and motion-energy notebooks (`fip_10`–`fip_17`, `men_00`–`men_05`). `TODO.md`
tracks the `kin_*` / `eph_*` work. History is in `CHANGELOG.md`. Last revised 2026-10-22 (ME normalization decided; ROI traces in the ME table; 2026-10-21: ME normalization items; 2026-10-09: isosbestic control added; 2026-10-08: notebooks
regrouped by analysis type; old names → new in `archive/README.md`).

---

## Plan: motion energy × DA/NE (talk for the brain-wide NM meeting)

Questions: how do DA and NE relate to movement; does RPE coding relate to movement; what movement
goes with DA/NE events; how are instructed and uninstructed movements represented. Outline A of the
2026-10-05 discussion, with the DA–NE correlation results added.

**Same choices everywhere.**
- Data: the CSV-curated assets (`fu.load_pairs`, 9 animals, 97 sessions) and the aligned ME table,
  **bottom camera only**, cameras that fail the video timing or quality screen left out
  (`inputs/video_screen_fip.csv`).
- DA, NE and ME on one 20 Hz grid (ME bin-averaged with `su.bin_to_grid`), each z-scored per
  session over the task.
- One ME onset rule (`fu.ME_ONSET_KW`), one transient rule (`fip_11` §1's prominence), one outcome
  window (0–2 s after the choice).
- Sessions averaged within animal; Wilcoxon signed-rank on animal means; circular-shift nulls
  (≥ 60 s) where chance is needed.
- Exclusions come from QC only. 816212 stays unless its clock check fails.

| # | Slide | Analysis | Where (2026-10-08 names) |
|---|---|---|---|
| 1 | Task and signals | none | |
| 2 | The ME data | QC counts, lick-triggered ME (clock check), ME around go cue and choice | `men_00` §0–1 |
| 3 | DA and NE co-vary | cross-correlation and coherence (+ DA–ME, NE–ME, by band); large transients with and without a partner | `fip_10` §1–2, `fip_11` §1 |
| 4 | Q1 movement | (a) DA, NE at ME onsets; (b) ME × DA and ME × NE cross-correlation; (c) variance explained by the task model and by task + lagged ME, model diagram, kernels | `fip_12` §1, `fip_10` §2, `fip_13` §1–3 |
| 5 | Movement and DA–NE coupling | DA–NE correlation raw, after the task model, after task + ME | `fip_13` §4 |
| 6 | Q2 RPE vs movement | (a) ME through the same RPE regression as DA and NE; (b) DA, NE RPE terms with and without post-outcome ME | `fip_16` §6 |
| 7 | Q3 movement at DA/NE events | ME around large DA and NE transients, solo vs partnered | `fip_11` §4 |
| 8 | Q4 instructed vs uninstructed | DA, NE, ME at instructed lick bouts, uninstructed lick bouts, ME events without licking | `fip_12` §2 |
| 9 | Summary | one line per question | |


Status 2026-10-08: everything in the table has run locally on 88 sessions / 9 mice and is in the talk deck (run then as `fip_07`, `fip_08`, `fip_09`; regrouped into `fip_10`–`fip_14` the same day and rerun). Not done: merging `fip-motion-energy` into `wild`.

Left out to keep it simple: the archived `fip_03`'s variance partition (`fip_13` §1 replaces it), slow engagement
coupling, sorting ME events by clustering or camera, timing claims from lags (GCaMP and dLight
kinetics differ).

---

## Open items

### Motion-energy normalization
Decided 2026-10-22 (`men_03`): ME traces, levels, responses, regressions and figures use floor ÷ SD
(max(0, ME − p1) / SD; default of `fu.add_motion_energy`, `ME_NORM = "floor_sd"` in the `fip_*` and `men_*`
notebooks); ME onsets stay on the z-scored trace; DA and NE stay z-scored.
- 808054 2025-09-05 is kept for now: its flicker is the strongest (floor 3× any other session's), and ÷ SD
  shrinks its ME responses by ~15% (SD after a 10 Hz low-pass / SD = 0.85; 0.98–0.99 elsewhere). Revisit if
  it drives a result.

### Aligned ME table: ROI traces (later)
aind-motion-energy 0.4.0 (2026-10-21) computes one trace per ROI next to the whole frame, from one ROI set
per batch. The table reads both 0.3.0 and 0.4.0 results (`me_results.py`) but stores only the whole-frame
`me_clean`, so nothing needs rebuilding until ROI traces are wanted.
- `build_me_table.py`: add one column per `roi_ids` entry from the 0.4.0 NPZ (`me_clean[:, k]`).
- `fu.load_me` (and `motion_energy_to_session`): an ROI argument, returning that column instead of the whole frame.

### Analyses
- `fip_11` §4 (Q3) by task context: partnered large transients fall mostly at outcomes and licking, so
  their higher ME may be context. Label each transient with `bu.label_context` (as `fip_11` §1 panel f) and
  compare ME around solo and partnered transients within each context.
- `fip_13` §4: why the DA–NE correlation rises from 0.02 (task removed) to 0.06 (task + ME removed).
  Correlate the ME-predicted parts of DA and NE; split the residual correlation by frequency band.
- Move `fip_16` (was `fip_05`) onto Rachel's code (agreed): `dummy_nwb.load`, her `data_z` / `data_z_norm`,
  `event_triggered_response`, her RPE slope with the outcome split and pre-first-reward trials
  dropped; keep in `fip_utils` only what the libraries don't have.
- Isosbestic control for the ME correlations (`fip_10` §2, `fip_12` §1, `fip_13`; asked for by
  Kyle Severson): is any of DA's or NE's relation to ME motion artifact left after motion correction?
  Do not use motion-corrected iso: the correction fits iso to itself, leaving ~0, so its ME
  correlation is weak whatever the correction did. Residual artifact is the part of a channel's
  artifact that iso does not contain. Instead:
  1. Raw iso (bleach-corrected `dff-bright` only, no `mc`) × ME through the same xcorr and
     ME-onset averages: how large the movement artifact is per fiber.
  2. Corrected DA/NE and raw iso averaged at ME onsets: an iso-shaped deflection (same timing and
     sign) left in corrected NE marks residual artifact.
  3. If the NE fiber's red channel carries no sensor, red × ME is an artifact measure the correction
     never used.
  4. Activity-independent recordings (GFP or a mutant sensor) through the same pipeline, if any exist.

  Data: the curated `df_fip.parquet` files carry only the corrected sensor channels (e.g. 816214
  2025-12-03: `G_1` NE, `G_3` DA, `dff-bright_mc-iso-IRLS`), with no iso or raw traces. Iso_1/Iso_3 (and
  red) have to come from upstream through Rachel's preprocessing without the `mc` step.
- Contralateral control: DA from the opposite NAc × NE.
- Behavior-model dependence: RPE and value come from one Q-learning fit; a refit or another model
  family would change `fip_16` sections 1–4.
- Indicator kinetics: deconvolve with published kernels, or compare with an nLight/GRAB-NE cohort.

---

## Closed

- ME normalization (2026-10-22): `men_03` compared the z-score, ÷ SD and floor ÷ SD. They share the session SD,
  so only ME levels read differently (`fip_11` §4); floor ÷ SD adopted for ME, the z-score kept for onsets
  and for DA/NE. `fip_10`–`fip_14`, `fip_16`, `men_00`–`men_03`, `men_05` rerun on it.
- `fip_04` and the older `DA_NE_4channels` asset (JSON curation): retired with `fip_00`–`fip_04`
  into `archive/` when the series was regrouped (2026-10-08); its open items (bilateral NAc pick,
  `no_fiber` channel, CSV curation API) go with it.
- Code Ocean runs and questions for Rachel: tracked outside this file (2026-10-08).
- Bout helpers into the video-analysis library: not now; they stay in `behavior_utils` (2026-10-08).
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
