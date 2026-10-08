# fip_* / men_* to-do

Open work for the FIP and motion-energy notebooks (`fip_00`–`fip_07`, `men_00`). `TODO.md` tracks
the `kin_*` / `eph_*` work. History is in `CHANGELOG.md`. Last revised 2026-10-05.

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
- One ME onset rule (`fu.ME_ONSET_KW`), one transient rule (`fip_07`'s prominence), one outcome
  window (0–2 s after the choice).
- Sessions averaged within animal; Wilcoxon signed-rank on animal means; circular-shift nulls
  (≥ 60 s) where chance is needed.
- Exclusions come from QC only. 816212 stays unless its clock check fails.

| # | Slide | Analysis | Where |
|---|---|---|---|
| 1 | Task and signals | none | |
| 2 | The ME data | QC counts, lick-triggered ME (clock check), ME around go cue and choice | `men_00` §0–1 |
| 3 | DA and NE co-vary | cross-correlation and coherence; large transients with and without a partner | `fip_07` Figs 1–2 |
| 4 | Q1 movement | (a) DA, NE at ME onsets; (b) ME × DA and ME × NE cross-correlation; (c) variance explained by the task model and by task + lagged ME | `fip_08` §1 |
| 5 | Movement and DA–NE coupling | DA–NE correlation raw, after the task model, after task + ME | `fip_08` §2 |
| 6 | Q2 RPE vs movement | (a) ME through the same RPE regression as DA and NE; (b) DA, NE RPE terms with and without post-outcome ME | `fip_08` §3 |
| 7 | Q3 movement at DA/NE events | ME around large DA and NE transients, solo vs partnered | `fip_08` §4 |
| 8 | Q4 instructed vs uninstructed | DA, NE, ME at instructed lick bouts, uninstructed lick bouts, ME events without licking | `fip_08` §5 |
| 9 | Summary | one line per question | |

Order: (1) local setup: data root from `ME_DATA_ROOT`, quality screen in `fu.me_sessions`,
`.venv-fip` kernel; (2) run `men_00`; (3) `fip_07` figures; (4) `fip_08`.

Status 2026-10-08: everything in the table has run locally on 88 sessions / 9 mice (`men_00`, `men_01` QC, `fip_07`, `fip_08` §0b–5 plus model figures, `fip_09` task-model variants) and is in the talk deck. Not done: merging `fip-motion-energy` into `wild`.

Left out to keep it simple: `fip_03`'s variance partition (4c replaces it), slow engagement
coupling, sorting ME events by clustering or camera, timing claims from lags (GCaMP and dLight
kinetics differ).

---

## Open items

### Analyses
- `fip_08` §4 (Q3) by task context: partnered large transients fall mostly at outcomes and licking, so
  their higher ME may be context. Label each transient with `bu.label_context` (as `fip_07` Fig 2f) and
  compare ME around solo and partnered transients within each context.
- `fip_08` §2: why the DA–NE correlation rises from 0.02 (task removed) to 0.06 (task + ME removed).
  Correlate the ME-predicted parts of DA and NE; split the residual correlation by frequency band.
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

---

## Closed

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
