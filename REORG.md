# REORG — kinematics_analysis (branch: wild)

Classification of `code/` for reorganization. Actionable ports live in `TODO.md`; this doc is the
current-state map only.

## Constraints

- **main is frozen + pushed + mirrored** in `AllenNeuralDynamics/LC-ephys-tonguemovements` →
  archiving on `wild` is fully reversible; nothing is at risk.
- **Import constraint:** notebooks import modules by bare name (`from data_loading import ...`).
  Active `.py` modules must stay flat in `code/`; only notebooks relocate (into `archive/`, where
  they don't need to run).
- No `eph_*`/`kin_*`/`fip_*` file references any archive-bound notebook (verified). The two cached
  intermediates new nbs read (`all_counts_df.parquet`, `filtered_ephys.pkl`) are rebuilt by the
  `eph_*` nbs themselves via `ephys_utils.build_all_counts_df`.

---

## KEEP — active analysis (flat)

- `eph_00`–`eph_06`, `eph_08` · `kin_00`–`kin_04` · `fip_00_explore`
- Modules: `data_loading.py`, `ephys_utils.py`, `encoding_methods.py`, `encoding_plots.py`,
  `per_unit_stats_registry.py`, `spatial_encoding.py`, `plotstyle.py`, `ccf_utils.py`
  - `per_unit_stats_registry.py`: distinct results-store + FDR + cross-analysis `compare`
    (used by `eph_01/02/03/04/06`); not superseded by `ephys_utils`/`encoding_methods`.
  - `ccf_utils.py`: imported by `eph_08` + `spatial_encoding.py`.

## KEEP — pipeline / data generation

- `build_all_tongue_movements.py` — builds `all_tongue_movements.parquet` (per-session
  `tongue_movs.parquet` → quality filter → concat). `%run` by `tongue_movements_all.ipynb`.
- `add_outbound.ipynb` — computes `out_*` outbound metrics, writes them into per-session parquets.
- `tongue_movements_all.ipynb` — builder wrapper + per-movement kinematic exploration (messy;
  holds a duplicate `compute_outbound_metrics`).
- `attach_data.ipynb` — batch-attaches data assets to the repo.
- Code Ocean data-gen: `run` → `run_batch_analysis.py`; plus `run_batch_analysis_modeltest.py`,
  `run_capsule.py`, `TransferToNWB.py` (video→NWB), `backup_nwb_utils_dynamicforaging.py`.
  Retained to generate data on future sessions.

## KEEP — model-quality / methods evaluation

- `pixel_error.ipynb` — keypoint-tracking pixel-error vs confidence (Lightning-Pose accuracy).
- `test_session_quality_analysis.ipynb` — session-level quality + session selection (feeds the
  `data_loading` inclusion filter).
- `test_session_wrapper.ipynb` — per-session analysis wrapper.
- `model_quality.ipynb` — foraging behavioral-model quality.
- `tongue_lickometer.ipynb` — lickometer↔Lightning-Pose lick-detection validation.

## HOLD — port unreplicated content before archiving (see `TODO.md`)

| File | Unreplicated content |
|---|---|
| `tongue_latency.ipynb` | RT+IMI decomposition; single-trial example fig; covert-preparatory narrative |
| `tongue_kinematics_ephys_intertrialmovs.ipynb` | within-trial vs ITI bout-aligned ephys encoding |
| `spatial_axis_comparison_rt_encoding_update.ipynb` | MERFISH (CCA) + retrograde (LDA) axis comparison + confidence cones (`eph_08` has waveform only) |
| `tongue_kinematics.ipynb` | lick↔movement correspondence (licks w/o movements, movements w/o licks, multi-lick) |
| `tongue_kinematics_cueresponse.ipynb` | cue-response spatial geometry (jaw/spout landmarks, endpoints by event) |

Also tracked in `TODO.md`: consolidate the duplicated `compute_outbound_metrics`
(`add_outbound` + `tongue_movements_all`) into the library.

## ARCHIVE — ready to move to `code/archive/`

Superseded by the new series / modules / library:
- `rt_ols_registry_pipeline.ipynb`, `registry_usage_example.py` → `ephys_utils` + `encoding_methods` + registry + `eph_01`
- `umap.ipynb` → `kin_03`
- `tongue_kinematics_ephys.ipynb`, `tongue_kinematics_ephys_figures.ipynb` → `eph_*` / `kin_04`
- `spatial_axis_comparison_rt_encoding.ipynb` (old dup of `_update`)
- `old_functions_tongue_kinematics.ipynb`, `cue_response_lick_processing_example.ipynb`,
  `tongue_segmentation_test.ipynb`, `batch_clips.ipynb`, `create_labeled_clip.ipynb`
  (functions promoted to / owned by the library)
- `example.ipynb`, `extract_tongue_kinematics.ipynb` (scratch)
- `event_timeline.ipynb` (task event-timeline plot)

Reference (→ `code/archive/reference/`):
- `F_ephys_behavior_action&outcome.ipynb` (collaborator's outcome/action auROC analysis)

---

## Execution plan

1. `git tag wild-prereorg` — restore point.
2. Create `code/archive/` (+ `code/archive/reference/`).
3. `git mv` the ARCHIVE files (history preserved; nothing imports them).
4. HOLD files stay until their `TODO.md` ports land, then archive.
