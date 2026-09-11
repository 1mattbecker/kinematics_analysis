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
  - `spatial_encoding.py`: `SpatialEncoder` — CCF maps, subgroup maps, spatial-dependence
    permutation tests. Contains **no axis-fitting machinery**; that is `spatial_axes.py`'s
    job (planned, see `TODO.md`).

### Planned additions (from the HOLD ports — see `TODO.md`)

Not yet written. Listed here so the target layout is legible before the ports land.

| Planned | Kind | Gates archiving of |
|---|---|---|
| `kin_02_latency` §8–§10 | extension | `tongue_latency` (with `kin_05`) |
| `kin_05_nonlick_movements.ipynb` | new | `tongue_latency`, `tongue_kinematics` |
| `kin_06_lick_geometry_choice.ipynb` | new | `tongue_kinematics_cueresponse` |
| `kin_07_value_encoding.ipynb` | new | `tongue_kinematics`, `tongue_kinematics_cueresponse` |
| `eph_07_bout_encoding.ipynb` | new | `tongue_kinematics_ephys_intertrialmovs` |
| `eph_09_structural_axes.ipynb` | new | `spatial_axis_comparison_rt_encoding_update` |
| `bout_utils.py` | new module | — (home for orphan `annotate_movement_bouts`) |
| `spatial_axes.py` | new module | — (`eph_08` refactors onto it) |

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

Audited cell-by-cell on 2026-09-11. The unreplicated content does **not** map one-notebook-to-one-port:
it clusters into five questions that cut across these files, with duplication between them. Read
the "Port plan for the five HOLD notebooks" overview at the top of `TODO.md` before starting any
of them.

| File | Unreplicated content | Ports into |
|---|---|---|
| `tongue_latency.ipynb` | RT + IMI decomposition (Δt de-shift, KS collapse, noise propagation w/ bootstrap CI); single-trial example fig; trial rasters by movement type / ordinal | `kin_02` §8–§10 **+** `kin_05` |
| `tongue_kinematics_ephys_intertrialmovs.ipynb` | within-trial vs ITI bout-aligned ephys encoding; sole copy of `annotate_movement_bouts` | `eph_07` **+** `bout_utils.py` |
| `spatial_axis_comparison_rt_encoding_update.ipynb` | RT-encoding spatial axis fit (`eph_08` skipped it), MERFISH (CCA) + retrograde (LDA) axes, bootstrap direction comparison, confidence cones | `eph_09` **+** `spatial_axes.py` |
| `tongue_kinematics.ipynb` | lick↔movement correspondence (licks w/o movements, movements w/o licks, multi-lick); per-trial non-lick structure; **kinematics vs behavioral-model latents** (Spearman/MI/RidgeCV/RF, prev-trial RPE) | `kin_05` **+** `kin_07` |
| `tongue_kinematics_cueresponse.ipynb` | jaw/spout landmark geometry + endpoints by event; **choice prediction from pre-lick kinematics** (ridge-logistic, AUC, binned P(right lick)); Q-value encoding | `kin_06` **+** `kin_07` |

Two corrections to the earlier reading of this table:

- **`cueresponse` is not just "spatial geometry."** The landmark figures are the setup for a
  choice-prediction analysis (cells 43–56) that is the actual result.
- **Kinematics × behavioral-model latents is an untracked topic**, present in *two* of these
  notebooks and in no `kin_*`/`eph_*` notebook. It gates two archivals, so neither
  `tongue_kinematics` nor `tongue_kinematics_cueresponse` can be retired without `kin_07`.

Already covered elsewhere — do **not** port: `tongue_kinematics` cells 78–79 (lick-detection
FP/FN parameter sweep) duplicate `tongue_lickometer.ipynb`, which is KEEP.

Also tracked in `TODO.md`: consolidate the duplicated `compute_outbound_metrics`
(`add_outbound` + `tongue_movements_all`) into the library. `annotate_movement_bouts` is the
same orphan-code pattern and is a candidate for the same treatment once `bout_utils.py` settles.

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

1. `git tag wild-prereorg` — restore point. — **done**
2. Create `code/archive/` (+ `code/archive/reference/`). — **done**
3. `git mv` the ARCHIVE files (history preserved; nothing imports them). — **done**
4. HOLD files stay until their `TODO.md` ports land, then archive.

Step 4 is the remaining work. Recommended order, by value ÷ risk (full rationale and
per-notebook section outlines are in `TODO.md`):

1. `kin_02` §8–§10 — pooled parquet only, no new dependencies, fully local-testable.
2. `kin_05_nonlick_movements` — mostly pooled; one Code Ocean-only section.
3. `eph_07_bout_encoding` + `bout_utils.py` — reuses `eph_00`'s raster/PETH helpers.
4. `spatial_axes.py` + `eph_09_structural_axes`, then refactor `eph_08` onto the module.
   Confirm the MERFISH / retrograde assets are reachable on Code Ocean **first**.
5. `kin_06_lick_geometry_choice`.
6. `kin_07_value_encoding` — last; new dependency on `get_mle_model_fitting`.

Archive a HOLD notebook only when **every** gate in the HOLD table above is met — three of the
five wait on two ports each.
