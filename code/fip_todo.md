# fip_* / men_* to-do

Open work for the FIP and motion-energy notebooks (`fip_10`–`fip_17`, `men_00`–`men_05`). `TODO.md`
tracks the `kin_*` / `eph_*` work; the isosbestic controls are in `fip_isosbestic_todo.md`. History is in `CHANGELOG.md`. Last revised 2026-11-16 (isosbestic controls split out; 2026-10-28: whisker-pad ROI table built and attached; next steps for ME; 2026-10-22: ME normalization decided; 2026-10-21: ME normalization items; 2026-10-09: isosbestic control added; 2026-10-08: notebooks
regrouped by analysis type; old names → new in `archive/README.md`).

---

## Moving to the DA/NE paper's shared repo (started 2026-11-16)

The ME and DA–NE analyses go into a shared repo with the paper's other authors: a Code Ocean capsule
that also runs locally once the final FIP asset is on open data (S3).

**What goes.**
- Core: `men_00`, `men_01`, `men_03`, `fip_10`, `fip_11`, `fip_12`, `fip_13`, `fip_16`.
- Optional: `fip_14`, `fip_15`, `fip_17`, `men_02`.
- Stays here (exploratory): `men_04`, `men_05`, `men_06`.
- Modules: `fip_utils`, `signal_utils`, `stats_utils`, `plot_utils`, `plotstyle`, `behavior_utils`,
  `men_utils`, `me_results`.
- ME table build (a `pipeline/` folder with the `inputs/` CSVs): `build_me_asset_map`,
  `build_me_table`, `check_leading_lost_frames`, `s3_utils`. `build_me_asset_map` imports
  `trigger_batch.make_client` from `../aind-motion-energy-batch-capsule/scripts`; ship
  `me_assets_fip.csv` as its output, or copy `make_client`.

**Done 2026-11-16.** One place for data locations (`fu.DATA_ROOT`, `CACHE_ROOT`, `FIG_ROOT`,
`fu.fip_asset_roots`, capsule layout `data/` + `scratch/`); no notebook or script carries paths;
`DANE_bilateral` (Rachel's combined asset, `4686d370`, test version) is used when attached; the
archived notebooks' NWB-list path removed from `fip_utils`. All 15 notebooks rerun: text outputs identical.
`inventory_pairs` reads parquet through pandas, so `fastparquet` alone is enough.

**Tested on the shared capsule's pins (2026-11-16).** The 12 core + optional notebooks ran locally on the
`da-ne-paper-figures` image's versions (rachel-analysis-utils `36a4615`, data-utils `v0.1.36`, basic-analysis
`improve_fip`, numpy 1.26.4, scipy 1.16.0, matplotlib 3.7.5, no pyarrow), caches rebuilt. No errors; text
outputs identical except four Wilcoxon p-values on lag measures (stepped values, so ties and zeros), all
far from 0.05: `fip_10` DA–NE peak lag 0.327 → 0.383 and DA–ME 0.484 → 0.547; `fip_11` §4 DA with NE
0.652 → 0.617, DA solo 0.652 → 0.664. Probably scipy's `wilcoxon` handling ties and zeros differently
(not checked against scipy's changelog). Means, SEMs and n unchanged. `scipy.signal.coherence` is ~6× slower in scipy 1.16
(45 vs 7.5 ms per call), so `fip_10` takes ~60 min instead of 13. Settle the shared image's scipy pin with
Rachel before quoting p-values.

**Next.**
1. When Rachel's final combined asset exists: add its folder name to `fu.FIP_ASSET_SETS` if it differs,
   attach it, rerun. Check how many of its 295 sessions are same-side DA/NE pairs (`fu.inventory_pairs`)
   and which have no ME (the ME table covers the current 97); update the notebooks' **Data** paragraphs.
2. Copy into the shared repo (proposed: `code/movement_da_ne/`, modules next to the notebooks); rename
   notebooks by question; agree the image's pins with Rachel (above); attach `DANE_bilateral` and the ME
   table (`e542022e`) to the capsule; decide where figures go (`fu.FIG_ROOT`; `results/` in a figures capsule?).
3. Local runs from open data: a download step for the FIP asset and the ME table into `data/`.

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

### Aligned ME table: ROI traces
aind-motion-energy 0.4.0 `rois_v2` run (2026-10-23, all sessions): whole frame on both cameras plus
`SideCameraRight_whiskerpad_box` on the side camera. Code done 2026-10-27 (branch `me-roi`): `build_me_table.py`
writes `me_clean_<roi id>` and `<camera>_roi_masks.npz`; `roi=` in `fu.load_me` / `motion_energy_to_session` /
`add_motion_energy` / `me_sessions`, `fu.me_rois`, `fu.load_roi_masks`; `"<camera>:<roi>"` sources in
`men_utils.load_session`. Tested on two sessions: whole-frame columns identical to the current table.
- Built locally 2026-10-27 from the 0.4.0 assets (`Code/data/me_table_rois_v2/`, 6.2 GB; the local
  `fip_motion_energy_aligned` symlink now points there, the old download `results-90c2d0a7…` is kept). Same 178 ok / 16
  refused; every existing column of all 178 cameras identical to asset `90c2d0a7`; all 90 ok side cameras carry the
  whisker-pad trace (64 after the video screen). Index wording follows library 0.5.0 (refused rows: empty `action`).
- Code Ocean: built in a workstation 2026-10-28 (`/scratch/fip_motion_energy_aligned_rois_v2`) and saved as asset
  `e542022e-7d56-48ec-a940-d2dcc7ec4733`, attached at `fip_motion_energy_aligned` in place of `90c2d0a7`.

**Where it stands (2026-10-28).** Work from branch `me-roi`. It does not wait for the library 0.5.0 kinematics
migration (separate handoff): no `fip_*` / `men_*` notebook or the modules they use imports what 0.5.0 removed, and the
table build and ME modules ran on 0.5.1. Only merging `me-roi` into `wild` waits, because it carries the 0.5.0 pin.

**Coverage.** Whisker pad: 64 sessions, bottom camera: 88; 62 have both. The video screen drops most side cameras
of two mice (overexposure): 809487 0 of 8, 818586 4 of 22. Whisker-pad results pool 8 mice, 7 with most of
their sessions:

| Mouse | 800886 | 808054 | 809487 | 809491 | 815334 | 816212 | 816214 | 818585 | 818586 |
|---|---|---|---|---|---|---|---|---|---|
| Bottom | 4 | 15 | 8 | 19 | 3 | 5 | 5 | 7 | 22 |
| Whisker pad | 4 | 15 | 0 | 19 | 3 | 7 | 5 | 7 | 4 |

**Next, in order:**
1. ~~Describe the whisker-pad trace~~ Done 2026-11-12: `men_06_whisker_pad_me` (run locally, 64 sessions, 8 mice).
   The box sits on the whisker pad in all 6 usable mask groups. The pad trace follows the whole frame (r 0.85 with
   the side, 0.88 with the bottom camera) with the same shape around the go cue, choice and licks, ~10–20% lower in
   session SDs. `floor_sd` behaves: stillness 0.05 SD above 0, steadier across the session than the whole-frame
   traces. Open: the pad's lick-triggered peak lag is +0.03 s (whole frame −0.02/−0.03 s), from a flat top over
   ±0.1 s around the lick; run `men_06` on Code Ocean.
   §5: pad ME events without licking (2.8/min, 39% of pad events, mostly quiet inter-trial) come with as much
   bottom- and side-camera ME as pad ME, so the box does not isolate whisking. 2026-11-16, §6–7: 70% of them have
   no bottom-camera event, but the bottom ME rises as much after those too. In 500 Hz spectra against still
   windows the pad gains more power at 16–30 Hz (8–15 Hz whisking, doubled because ME is unsigned) than at
   2–8 Hz, in all 8 mice and more than either whole frame (p = 0.008 / 0.016): a whisking signature in the
   level-matched pad trace, though not in its event timing. 808054's bottom camera has an imaging line at ~28 Hz.
   Not yet done: a smaller box on the whisker shafts in front of the snout (new ROI, ME rebuild);
   whisker tracking.
2. `men_04`'s go-cue items on the whisker pad: 500 Hz whisker-pad ME in the first 150 ms. Is the 10–16 ms jump and the
   ~24 / ~55 ms peaks there, and in which of the five "jump" and four "no jump" mice? This tests the startle-plus-whisking
   reading that so far rests on the maps.
3. If it holds up, a second ME signal in the FIP analyses: `fu.add_motion_energy(..., camera="SideCameraRight",
   roi="whiskerpad_box")` next to the bottom camera in `fip_12` (DA/NE at ME onsets) and `fip_13` (task + ME regression).

### `men_04`: ME in the first 150 ms after the go cue
Found so far: five mice jump 10–16 ms after the cue, four don't; the high-passed ME peaks at ~24 and
~55 ms, on the whiskers (maps).
- Is the ~24 ms peak real in the mice without the jump? It falls where their movement starts, and the
  15 Hz high-pass turns a fast rise into a peak there. Pass a ripple-free rise shaped like theirs
  through the same filter and see whether it makes a ~24 ms peak. (The ~55 ms peak is clear of this.)
- Startle plus whisking is inferred from the maps only. Look at the movement: a slowed-down,
  cue-aligned clip of a few trials, or whisker-angle tracking.
- Why five mice jump and four don't: check rig, cohort and go-cue sound settings against the split.
- The maps (where the motion is) come from one session per mouse; repeat on a second session per mouse.

### Analyses
- `fip_11` §4 (Q3) by task context: partnered large transients fall mostly at outcomes and licking, so
  their higher ME may be context. Label each transient with `bu.label_context` (as `fip_11` §1 panel f) and
  compare ME around solo and partnered transients within each context.
- `fip_13` §4: why the DA–NE correlation rises from 0.02 (task removed) to 0.06 (task + ME removed).
  Correlate the ME-predicted parts of DA and NE; split the residual correlation by frequency band.
- Move `fip_16` (was `fip_05`) onto Rachel's code (agreed): `dummy_nwb.load`, her `data_z` / `data_z_norm`,
  `event_triggered_response`, her RPE slope with the outcome split and pre-first-reward trials
  dropped; keep in `fip_utils` only what the libraries don't have.
- Isosbestic control for the ME correlations: moved to `fip_isosbestic_todo.md` (2026-11-16).
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
