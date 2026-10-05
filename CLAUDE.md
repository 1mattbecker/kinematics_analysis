# CLAUDE.md — kinematics_analysis

This file tells Claude Code about this project. Read it at the start of every session.

---

## Project overview

Tongue kinematics analysis pipeline for dynamic foraging behavior videos in head-fixed mice.
Part of a research program studying LC-NE and catecholamine control of psychomotor behavior
at the Allen Institute for Neural Dynamics (AIND).

Primary analysis environment is Code Ocean (cloud). This local repo is for development
and code editing. Data lives on Code Ocean — do not expect data files to be present locally.

---

## Rules

- **Python 3.12.** Every active branch (`wild`, `main`, `kinematics-manuscript`) runs the
  Python 3.12 Code Ocean image as of 2026-09-25, so 3.10+ syntax (`X | None`, `match`) is fine
  in this repo's own code. Package versions are fixed by `environment/constraints.txt`
  (pandas stays 2.x until the NWB stack moves). Change a version there, deliberately, to
  upgrade it.
  Code headed for the library (`aind-dynamic-foraging-behavior-video-analysis`) must still
  run on Python 3.11.
- **Do not modify anything in `/environment`.** That folder controls the Code Ocean
  Docker build and should only be changed intentionally.
- **Do not modify `.codeocean/` config files.**
- **Branch behavior depends on which branch is active:**
  - `wild` — **development with Claude.** Agentic work with latitude: larger changes are
    acceptable, but commit frequently and summarize what changed.
  - `main` — **verified, working code only.** It gets there by merging `wild` into `main`
    once the changes have been run and checked. Do not commit to or push `main` unless the
    user explicitly asks for that promotion.
  - `kinematics-manuscript` — forward-looking manuscript work; periodically takes `wild`.
  - Retired, kept as tags (not branches): `archive/local-dev` (old careful-dev branch) and
    `archive/main-py39` (main before its 2026-09-25 promotion to `wild`; Python 3.9 image,
    with AIND libraries pinned so it still builds). Restore one with
    `git switch -c <name> archive/<tag>`.
- **Preserve existing function signatures** unless explicitly told to change them.
  Other notebooks may depend on them.
- **Do not delete or overwrite data loading cells** in notebooks — data paths are
  Code Ocean-specific and will differ locally.

---

## Repository structure

```
kinematics_analysis/
├── code/               # All analysis code — notebooks + flat modules
│   └── archive/        # Superseded/scratch notebooks (provenance only, not run)
│       └── reference/  # Collaborator notebooks kept for reference
├── data/for_local/     # Small local subset for development (see Data below)
├── inputs/            # Small input tables scripts read (session lists, asset maps); see its README
├── environment/        # Docker + postinstall scripts (DO NOT MODIFY)
├── metadata/           # Project metadata
├── .codeocean/         # Code Ocean config (DO NOT MODIFY)
├── CLAUDE.md           # This file
├── TODO.md             # Actionable work — read before starting anything
└── CHANGELOG.md        # Notable changes, newest first
```

**Local FIP environment:** `.venv-fip` (Python 3.12, created with `uv`) holds `rachel-analysis-utils@864550d`
(the Dockerfile pin), `aind-dynamic-foraging-data-utils@32e8dbe`, `aind-dynamic-foraging-basic-analysis@8ec194f`
and `aind_analysis_arch_result_access`, with versions held by the constraints file (built from `environment/py39-constraints.txt`, before the
2026-09-28 package upgrade; rebuild it against `environment/constraints.txt` to match the capsule). Use it
for `fip_*` notebooks that call Rachel's code locally. The older `.venv` (3.9) cannot install her package.

**Import constraint:** notebooks import repo modules by bare name
(`from data_loading import ...`), so active `.py` modules must stay **flat in `code/`**.
Only notebooks relocate into `archive/`, where they don't need to run.

---

## The library — `aind-dynamic-foraging-behavior-video-analysis`

Most domain code lives in an installable AIND library, not in this repo. Wherever `TODO.md`
says "the library", it means this one.

- **Repo:** `AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis`, default
  branch `main`. **Local clone:** `../aind-dynamic-foraging-behavior-video-analysis`.
- **Installed on Code Ocean** by `environment/Dockerfile` as an editable git checkout pinned
  to a commit: tag `v0.2.0` (`0e1c8df`, 2026-10-05): the video timing QC (added in `v0.1.0`,
  `41e5b59`) with `timing_verdict`, and video screening. Library
  changes reach the capsule only when that pin is moved deliberately, then the image rebuilt.
  To go back to the version before the QC, pin tag `pre-video-timing-qc` (`5738b32`). The QC
  corrects `time_raw` for drop/glitch sessions and refuses Harp clock-step sessions; see the
  library's `VIDEO_TIMING_QC_PLAN.md`.
- **Import path:** `aind_dynamic_foraging_behavior_video_analysis`. `requires-python = ">=3.9"`
  today; it will move to `">=3.11"` once every consumer capsule and branch is on 3.11+ (the
  library's `PYTHON_311_UPGRADE_PLAN.md`, Stage 3). This capsule's `wild` branch runs 3.12.
- Its `README.md` is the AIND template plus a "Scope" section stating what belongs in the
  library — read that section, then the module source.

| Library module | What it owns |
|---|---|
| `kinematics/tongue_kinematics_utils.py` | Keypoint I/O and masking, filtering, `segment_movements`, `aggregate_tongue_movements` (emits the `out_*` outbound metrics natively via `compute_outbound_metrics`), `annotate_trials_*`, `assign_movements_to_licks`, `get_trial_level_df`, `filter_timestamps_refractory` |
| `kinematics/tongue_analysis.py` | Batch driver — `run_batch_analysis`, `generate_tongue_dfs`, `analyze_tongue_movement_quality`; declares the `tongue_quality_stats.json` contract (`TONGUE_QUALITY_STATS_FILENAME`, `load_tongue_quality_stats`, `get_quality_summary`) |
| `kinematics/tongue_lickometer_utils.py` | Spout-*contact* lick detection and event scoring — `detect_licks`, `calculate_metrics`, `calculate_metrics_witheventkeys`; used by `val_02`/`val_03` |
| `kinematics/video_clip_utils.py` | Clip extraction (`extract_clips_ffmpeg_encode` = frame-accurate re-encode, `extract_clips_ffmpeg_after_reencode` = fast `-c copy`) + labeling, `find_labeled_video`, `get_video_time` |
| `kinematics/kinematics_nwb_utils.py` | Session-ID parsing, NWB file lookup |
| `ephys/tongue_ephys.py` | Raster/PETH machinery — `make_rp_and_events`, `compute_psth`, `RasterPlotter`, `load_intermediate_data`, `get_session_prefix` |
| `video_alignment.py` | Video↔session/behavior clock conversion — `compute_video_session_offset`, `session_time_to_video_time` |
| `TransferToNWB.py` | Bonsai JSON/mat → NWB (a copy also sits in `code/`) |

### Library vs repo boundary

Decided 2026-09-17 (the library records the same rule in its `README.md`, "Scope").
One test: **would another AIND project doing tongue kinematics want this, unchanged?**

- **Library:** code that produces or annotates the per-session intermediates
  (`tongue_kins.parquet`, `tongue_movs.parquet`, `kps_raw_*.parquet`,
  `tongue_quality_stats.json`), runs in the batch pipeline, or is generic to tongue-kinematics
  sessions — keypoint I/O and filtering, segmentation, aggregation (including `out_*`),
  trial/lick annotation, QC stats, lick detection, video/NWB lookup, clip extraction,
  raster/PSTH primitives. It must stay stable: other consumers take `main` when they move their pins.
- **This repo:** analysis built *on top of* the intermediates for the LC-NE RT-encoding
  question — encoding models, per-unit registries, spatial topography and axes, figure style.
  Free to churn.
- **Plots:** the library's plot functions are pipeline QC artefacts written to disk by
  `analyze_tongue_movement_quality`. They carry no styling contract and are not used for
  figures shown in notebooks here; presentation plotting is this repo's job (`plotstyle.py`,
  `plot_utils.py`, `encoding_plots.py`). Do not restyle the library's plotters and do not add plotters there.
- **Contracts:** a file the library writes and this repo reads is declared next to the writer
  and read through the accessor, never by path — `tongue_quality_stats.json` is read via
  `load_tongue_quality_stats` / `get_quality_summary` in `data_loading.py` and
  `build_all_tongue_movements.py`.
- **Ambiguous cases** stay here until a second consumer appears: `build_trial_features` (its
  column set is chosen for this project's encoding models), `kin_06`'s landmark plot. The bout
  helpers (`annotate_movement_bouts`, `classify_bout_times`) got their second consumer (`men_00`)
  and moved to `behavior_utils.py`; they are library candidates (a library PR).
- **Layering** is written into the module docstrings on both sides
  (`tongue_kinematics_utils`, `tongue_lickometer_utils`, `tongue_ephys` in the library;
  `ephys_utils` here). Library changes go on a branch of the library repo as a PR; the capsule
  picks them up on the next image rebuild, so batch library changes and rebuild once.

---

## Key notebooks and scripts

*(Update this section as files are added or renamed.)*

Analysis notebooks are numbered series, each a linear line of questioning on one topic.
Many are **Code Ocean-only** — they use per-session intermediates absent from `data/for_local/`
and guard those sections behind an `IS_CO` skip so they still run clean locally.

### `kin_*` — tongue kinematics (behavior only)

| Notebook | Question |
|---|---|
| `kin_00_movement_qc` | What is noise vs. real tongue movement? (See the noise-definition note below.) |
| `kin_01_population` | How are movement parameters distributed across sessions? |
| `kin_02_latency` | How does latency vary with cue-response ordinal k, and does it decompose as RT ≈ RT₁ + (k−1)·Δt? |
| `kin_03_umap` | Discrete clusters or continuous structure in movement kinematics? |
| `kin_04_timecourse` | Does movement vigor drift across a session, or track reward? |
| `kin_05_nonlick_movements` | Do the lickometer and video streams describe the same events — and what are the movements that aren't licks? |
| `kin_06_lick_geometry_choice` | Does the direction of a preparatory movement predict which spout is licked? |
| `kin_07_value_encoding` | Do behavioral-model latents (Q, RPE) show up in *how* the tongue moves? |

### `eph_*` — LC units × tongue kinematics

| Notebook | Question |
|---|---|
| `eph_00_single_unit_inspection` | Per-unit rasters/PETHs — the visual entry point |
| `eph_01_monovariate_rt` | Does reaction time predict spike count? |
| `eph_02_monovariate_kinematics` | Does spike count predict tongue kinematics? |
| `eph_03_partial_correlations` | Is kinematics encoding RT-independent, or shared variance? |
| `eph_04_spatial` | Where in CCF do RT and kinematics encoders cluster? |
| `eph_05_temporal` | When, relative to the go cue, does RT-predictive firing occur? |
| `eph_06_behavioral_comparison` | RT encoding vs. the AIND behavioral model ("Sue") outputs |
| `eph_07_bout_encoding` | Do units respond differently to within-trial vs. ITI movement bouts? |
| `eph_08_waveform_axis` | Poster figure: \|T_rt\| vs. each unit's projection onto the waveform axis |
| `eph_09_structural_axes` | Does the RT-encoding gradient align with waveform / MERFISH / projection-target axes? |

`eph_01`–`eph_04` and `eph_06` all write into `PerUnitStatsRegistry`, so their per-unit
T-statistics compose and can be compared across analyses. Add new per-unit measures through
`encoding_methods`/the registry rather than inline, so they compose too.

### `fip_*` — fiber photometry

| Notebook | Question |
|---|---|
| `fip_00_explore` | FIP access/plotting; relate signals to motion energy, single-session then pooled |
| `fip_01_movement_value_coding` | Motion energy × RPE / value coding (tonic value, phasic RPE) |
| `fip_02_ne_only_events` | Do NE and DA transients dissociate around movement onsets? |
| `fip_03_da_ne_commonality` | How much of the motion-energy variance DA and NE explain is unique to each and how much is shared? |
| `fip_04_da_ne_xcorr` | Do DA and NE co-vary across animals? Same-hemisphere DA × NE cross-correlation and coherence over every curated session of both FIP assets (`DA_NE_4channels`, `DANE_3channels_curated`), with animal as the unit |
| `fip_05_da_ne_rpe_coupling` | How are the DA and NE (LC-axon) RPE signals related? In order: RPE coding in each and how alike their outcome/RPE coefficients are (trial, session, animal), trial-by-trial correlation of outcome responses (total, within outcome, residual after outcome + RPE, with response time / trial position / baselines as separate controls), timing/magnitude of phasic transients, tonic/baseline coupling by timescale (raw vs task-residual), and how often each has a large transient without the other. Reads the CSV-curated parquet assets directly (`fip_utils.load_pairs`), so it also runs locally |
| `fip_07_da_ne_summary` | Summary of `fip_04` + `fip_05` on the CSV-curated assets, one set of conventions (+ lag = NE later; DA vermillion, NE blue, rewarded orange, unrewarded grey). Fig 1 xcorr/coherence, Fig 2 large transients with a schematic, Fig 3 task responses via Rachel's pipeline (`dummy_nwb`, `get_average_signal_window`, `event_triggered_response`, `Qch-binned3`), drawn once per window (Rachel 0.33–1 s; early/late/full). Fig 3 inputs cached (~15 min first build) |
| `fip_06_rpe_split` | Rachel's RPE slopes split trials by RPE sign; splitting by outcome differs only for unrewarded trials with Q_chosen = 0 (1%). Where they come from (forget rate fit at 1.0; before the first reward), what they look like, and how moving them changes the slopes. Runs on `.venv-fip` with Rachel's functions |
| `fip_08_movement_da_ne` | How do DA and NE relate to movement (motion energy, bottom camera, quality-screened)? ME onsets and DA/NE × ME cross-correlation; variance explained by the task model vs lagged ME; the DA–NE correlation with task and ME removed; RPE terms of DA, NE and ME and with ME as a covariate; ME at large solo vs partnered transients; instructed vs uninstructed lick bouts and movement without licking. Runs locally on `.venv-fip` (kernel `kinematics_fip`) given `ME_DATA_ROOT`; ~25 min |

### `men_*` — motion energy (behavior only)

| Notebook | Question |
|---|---|
| `men_00_motion_energy_description` | What does motion energy (bottom and side cameras, aligned ME table) look like against the task? Lick-triggered clock check; trial averages by go cue and choice; rewarded vs unrewarded; responded vs no response; instructed (≤2 s after go cue) vs uninstructed lick bouts; ME events with vs without licking against a shifted-lick chance level; running mean over the session; reward/failure streaks. Reads trials/licks from the CSV-curated FIP assets (no photometry). Runs locally on `.venv-fip` given `ME_DATA_ROOT` |

### `val_*` — detection validation

| Notebook | Question |
|---|---|
| `val_02_lickometer` | Can Lightning-Pose detect a lick *defined as tongue–spout contact*? The repo's only measure of **precision** (the library's `coverage_pct` path is recall-only) |
| `val_03_missed_licks` | The converse — when the lickometer misses a lick, does pose tracking see it? |
| `val_04_lickometer_qc` | The answer to #96 — per session, what fraction of *contact-like* tongue excursions (calibrated against that session's lickometer-confirmed contacts) have no lickometer event, split by lickometer context (skipped beat / silent bout / bout edge / isolated), with gates and flags. Runs locally on the exported event tables in `data/for_local/` |
| `val_05_lickometer_qc_methods` | The analyses behind every choice in `val_04`: where the raw pose/lickometer disagreement comes from (pre-trial, disengaged tail, keypoint jitter at the spout), depth/dwell of confirmed vs unconfirmed excursions, matching (onset vs closest approach, window, one-to-one vs any-in-window, alignment), the skipped-beat test, sensitivity of the ranking. `val_03` is the earlier long-form workup |

### Model quality / methods evaluation

- `pixel_error.ipynb` — keypoint-tracking pixel error vs. confidence (Lightning-Pose accuracy)
- `test_session_quality_analysis.ipynb` — session-level quality and session selection; feeds
  the `data_loading` inclusion filter
- `model_quality.ipynb` — a single-session **Lightning-Pose pipeline walkthrough** (load →
  mask `tongue_tip_center` @0.90 → filter → segment → annotate → lick coverage), already on
  modern library imports. Despite the name it is not about foraging behavioral-model quality

### Pipeline / data generation

`build_all_tongue_movements.py` (per-session `tongue_movs.parquet` → quality filter →
`all_tongue_movements.parquet`; `%run` by `tongue_movements_all.ipynb`) · `add_outbound.ipynb`
(computes `out_*` outbound metrics into per-session parquets) · `attach_data.ipynb` ·
`test_session_wrapper.ipynb` (batch runner wrapping `run_batch_analysis`; mostly commented
out — operations, not evaluation) · `run_batch_analysis.py` / `run_capsule.py` /
`TransferToNWB.py` / `backup_nwb_utils_dynamicforaging.py` · `build_me_table.py` (aligned
motion-energy table for the FIP sessions, run in a cloud workstation; `build_me_asset_map.py`
maps sessions to ME assets, `check_leading_lost_frames.py` is the lick-triggered-ME timing check;
all three read S3 through `s3_utils.py`; see `fip_me_aligned_table_plan.md`).

### Repo modules (`code/*.py`, flat by necessity)

| Module | What it owns |
|---|---|
| `signal_utils.py` | Generic time series (numpy/scipy, no domain code), imported as `su`: `zscore`, `threshold_onsets` (run length in samples or s), `bin_to_grid`, grid `peri_event_grid` / `window_mean_grid`, `rolling_mean`, `norm_xcorr` / `circular_xcorr` (**one lag convention: `norm_xcorr(a, b)` peaks at + lag when `a` is later**), `xcorr_shift_null` / `coherence_shift_null` / `band_corr_with_null`, `detect_transients`, `nearest_partner`, `near_any`, `coincidence_null` |
| `stats_utils.py` | Pooling and tests, imported as `st`: `mean_sem` (ddof=1, per-point n), `group_means` (sessions → animals), `animal_means`, `wilcoxon_animals`, `corr`, `shift_p` / `shift_z`, `cluster_test`, `hier_bootstrap` (wraps `aind_hierarchical_bootstrap`; session-weighted), `fdr_bh` / `wilson_ci` (statsmodels), `stars`, `fmt_p` |
| `plot_utils.py` | Summary plots, imported as `pu`: `plot_mean_sem`, `plot_etr` (tidy ETR → mean ± SEM), `strip_by_measure`, `strip_by_animal`, `animal_styles`. Style itself stays in `plotstyle.py` |
| `behavior_utils.py` | Task behaviour shared across series, imported as `bu`: `lick_bouts` (library `annotate_lick_bouts`, 0.7 s gap), `annotate_movement_bouts` / `classify_bout_times` / `get_session_bout_times`, `label_context` (rewarded / unrewarded / cue-no-response / licking / quiet) |
| `data_loading.py` | Session/unit QC and inclusion filters, `load_units_with_spike_times` |
| `ephys_utils.py` | `AnalysisConfig`, spike counting (`count_spikes`, `count_spikes_in_window`), `session_offset`, `load_example_session_and_unit`, session bundles, `build_all_counts_df`, `build_trial_features` |
| `encoding_methods.py` | Generic per-unit encoding — `AnalysisSpec`, `fit_encoding`, OLS/GLM/Spearman/partial |
| `per_unit_stats_registry.py` | Results store + FDR + cross-analysis `compare`; used by `eph_01/02/03/04/06` |
| `encoding_plots.py` | Stateless plot functions for encoding results |
| `spatial_encoding.py` | `SpatialEncoder` — CCF maps, spatial-dependence permutation tests. **No axis fitting** |
| `spatial_axes.py` | Fitting/comparing 3-D gradient *directions* — linear/CCA/LDA + bootstrap, `compare_bootstrap_directions`, `cone_half_angle`. Coordinate-frame agnostic (takes plain Nx3) |
| `ccf_utils.py` | CCF conversions — `pir_to_lps`, `ccf_pts_convert_to_mm`, `project_to_plane` |
| `plotstyle.py` | Figure standards — `apply_style`, `style_ax`, `save_fig`, Okabe-Ito colors |
| `kin_utils.py` | `kin_*` helpers (numpy/pandas only): `coerce_bool`, `load_kps_raw`, `jaw_position` |
| `lickometer_qc.py` | Shared machinery for `val_04`/`val_05` — builds the pose-excursion and lickometer event tables from intermediates (`build_event_tables`, CO only), scores excursions against the lickometer (`annotate_pose_events`), calibrates "contact-like" per session (`contact_reference`), labels candidates and their context, and reduces to one row per session (`summarize_sessions`). numpy/pandas only; the library is imported lazily |
| `fip_utils.py` | All FIP code (imported as `fu`). NWB-list path for `fip_00`–`fip_04`: `load_curated_sessions` (JSON curation; `use_curation=False` for assets curated at build time), `select_session`, `build_meta`, `pick_example`, `enrich_trials` (Rachel's `enrich_df_trials`), `process_session`. CSV-curated path for `fip_05`/`fip_07`/`men_00` (was `fip_coupling.py`): `find_asset_root`, `inventory_pairs`, `choose_side`, `load_pairs`, `trial_measures`, `fit_rpe_terms`, `residualize`, `task_residuals`. Motion energy on the FIP clock from the aligned ME table: `load_me`, `me_sessions`, `motion_energy_to_session` (cut to the task ± 30 s), `ME_ONSET_KW` (the one ME onset rule). Does *not* own the choice of FIP normalization — which `enrich_dfs` call a notebook runs stays visible in that notebook |
| `men_utils.py` | `men_*` loading: `inventory_sessions` (ME table × trial/event parquets), `load_sessions` (ME bin-averaged onto a 100 Hz session grid, lick events kept), `normalise` |
| `s3_utils.py` | Public-S3 reads over HTTPS for the ME scripts (standard library only): `urlopen` with retries, `download`, `fetch_cached`, `read_json`, `list_keys` |

`spatial_encoding.py` and `spatial_axes.py` are deliberately separate: *where* a statistic is
large and *in which direction* it changes are different questions with different inputs.

**Before writing a helper, look for one.** Check the libraries first (basic-analysis, data-utils,
Rachel's utils, `aind_hierarchical_bootstrap`, the video-analysis library, statsmodels), then
the modules above. If a near match exists, extend it with an optional argument instead of
copying it. Code needed by two or more notebooks goes in a module, not inline.

### Things worth knowing before editing

- **Cached intermediates:** exactly two are cached (`all_counts_df.parquet`,
  `filtered_ephys.pkl`), and the `eph_*` notebooks rebuild them via
  `ephys_utils.build_all_counts_df`. Results stay in memory; reader notebooks re-fit.
- **Spike-count windows are load-bearing.** `eph_08`/`eph_09` use
  `count_window_s=(0.0, 0.5)`, `baseline_window_s=(-2.0, 0.0)`, matching the paper and the
  poster figure. Changing them silently breaks the replication — see `TODO.md`.
- **Column-name trap:** in `all_tongue_movements_*.parquet`, `endpoint_x/y` and
  `max_x_from_jaw`/`max_y_from_jaw` are **absolute camera pixel positions** despite the names;
  the jaw-relative distances are `max_x_distance`/`max_y_distance`. They are not pooled-safe
  raw — each session carries its own camera/jaw offset. Use `kin_06`'s `get_jaw_positions()`.
- **QC "noise"** means confident misdetection of the wrong body part. Lickometer agreement,
  confidence, and duration are *not* valid noise filters — use cross-keypoint geometry and
  trajectory shape.

---

## Data

### Local test data
A subset of data is available locally for development and testing in:
`data/for_local/`

| File | Size | Contents |
|---|---|---|
| `all_tongue_movements_04022026.parquet` | 66.9 MB | Main kinematics dataset — tongue movement bouts |
| `features_combined_beh_all.pkl` | 2.9 MB | Combined behavioral features |
| `filtered_ephys.pkl` | 2.5 MB | Filtered electrophysiology data |
| `all_counts_df.parquet` | 2.7 MB | Count data |
| `20250418_transformed_remesh_10_ccf25.obj` | 3.2 MB | CCF brain mesh (3D) |
| `new_core_mesh.obj` | 1.7 MB | Core brain mesh (3D) |

When testing code locally, use these files. Adjust paths accordingly:
- Local: `data/for_local/filename`
- Code Ocean: `/root/capsule/data/...`

### Code Ocean data (not available locally)
Primary reference file on Code Ocean:
`/root/capsule/data/LCrecordings_combined_units/combined_unit_tbl.pkl`

If code cannot run locally due to missing data, note the expected input schema
and write/edit the logic without executing it.

---

## Code style

- NumPy-style docstrings for new functions
- Add a short comment when adding new metrics or features
- Keep notebook cells modular — one logical step per cell
- Prefer explicit variable names over abbreviations

---

## Git workflow

- Commit frequently with descriptive messages before and after significant changes
- Format: `git commit -m "what changed and why"`
- After any agentic task, summarize what files were changed and why
- Record notable changes in `CHANGELOG.md` (newest first, dated `YYYY-MM-DD`); add a dated
  section at the top for new work

---

## When in doubt

Ask before making broad changes. Prefer targeted, minimal edits over restructuring.
This codebase runs on Code Ocean — local runnability is secondary to correctness.
