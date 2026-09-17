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

- **Python 3.9 compatible syntax only.** No `|` for type unions, no structural pattern
  matching, no walrus operator in complex contexts. This matches the Code Ocean environment.
- **Do not modify anything in `/environment`.** That folder controls the Code Ocean
  Docker build and should only be changed intentionally.
- **Do not modify `.codeocean/` config files.**
- **Branch behavior depends on which branch is active:**
  - `local-dev` — careful development with user oversight. Make targeted, minimal edits.
    Always show diffs and wait for approval before applying broad changes.
  - `wild` — agentic refactoring with more latitude. Larger changes are acceptable,
    but still commit frequently and summarize what changed.
  - `main` — do not modify or push to main under any circumstances.
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
├── environment/        # Docker + postinstall scripts (DO NOT MODIFY)
├── metadata/           # Project metadata
├── .codeocean/         # Code Ocean config (DO NOT MODIFY)
├── CLAUDE.md           # This file
├── TODO.md             # Actionable work — read before starting anything
└── CHANGELOG.md        # Notable changes, newest first
```

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
  to `@main` — so anything merged to `main` reaches the capsule on its next image build.
  There is no version pin to shield against upstream changes.
- **Import path:** `aind_dynamic_foraging_behavior_video_analysis`. `requires-python = ">=3.9"`,
  which is what keeps it compatible with this capsule.
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
  raster/PSTH primitives. It must stay stable: the capsule installs it from `main` unpinned.
- **This repo:** analysis built *on top of* the intermediates for the LC-NE RT-encoding
  question — encoding models, per-unit registries, spatial topography and axes, figure style.
  Free to churn.
- **Plots:** the library's plot functions are pipeline QC artefacts written to disk by
  `analyze_tongue_movement_quality`. They carry no styling contract and are not used for
  figures shown in notebooks here; presentation plotting is this repo's job (`plotstyle.py`,
  `encoding_plots.py`). Do not restyle the library's plotters and do not add plotters there.
- **Contracts:** a file the library writes and this repo reads is declared next to the writer
  and read through the accessor, never by path — `tongue_quality_stats.json` is read via
  `load_tongue_quality_stats` / `get_quality_summary` in `data_loading.py` and
  `build_all_tongue_movements.py`.
- **Ambiguous cases** stay here until a second consumer appears: `annotate_movement_bouts` /
  `classify_bout_times` in `ephys_utils.py`, `build_trial_features` (its column set is chosen
  for this project's encoding models), `kin_06`'s landmark plot.
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
`TransferToNWB.py` / `backup_nwb_utils_dynamicforaging.py`.

### Repo modules (`code/*.py`, flat by necessity)

| Module | What it owns |
|---|---|
| `data_loading.py` | Session/unit QC and inclusion filters, `load_units_with_spike_times` |
| `ephys_utils.py` | `AnalysisConfig`, spike counting, session bundles, `build_all_counts_df` — **and** behavior-derived features that serve ephys alignment (`build_trial_features`, `annotate_movement_bouts`, `classify_bout_times`) |
| `encoding_methods.py` | Generic per-unit encoding — `AnalysisSpec`, `fit_encoding`, OLS/GLM/Spearman/partial |
| `per_unit_stats_registry.py` | Results store + FDR + cross-analysis `compare`; used by `eph_01/02/03/04/06` |
| `encoding_plots.py` | Stateless plot functions for encoding results |
| `spatial_encoding.py` | `SpatialEncoder` — CCF maps, spatial-dependence permutation tests. **No axis fitting** |
| `spatial_axes.py` | Fitting/comparing 3-D gradient *directions* — linear/CCA/LDA + bootstrap, `compare_bootstrap_directions`, `cone_half_angle`. Coordinate-frame agnostic (takes plain Nx3) |
| `ccf_utils.py` | CCF conversions — `pir_to_lps`, `ccf_pts_convert_to_mm`, `project_to_plane` |
| `plotstyle.py` | Figure standards — `apply_style`, `style_ax`, `save_fig`, Okabe-Ito colors |
| `lickometer_qc.py` | Shared machinery for `val_04`/`val_05` — builds the pose-excursion and lickometer event tables from intermediates (`build_event_tables`, CO only), scores excursions against the lickometer (`annotate_pose_events`), calibrates "contact-like" per session (`contact_reference`), labels candidates and their context, and reduces to one row per session (`summarize_sessions`). numpy/pandas only; the library is imported lazily |
| `fip_utils.py` | Shared setup for the whole `fip_*` series (imported as `fu`): curation/loading (`load_curated_sessions`), `parse_event`/`get_trace`/`build_meta`/`pick_example`, trial enrichment, motion energy on the FIP clock, signal helpers, `process_session`. Deliberately does *not* own the choice of FIP normalization — which `enrich_dfs` call a notebook runs stays visible in that notebook. See `code/fip_todo.md` for a pending Dockerfile pin |

`spatial_encoding.py` and `spatial_axes.py` are deliberately separate: *where* a statistic is
large and *in which direction* it changes are different questions with different inputs.

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
