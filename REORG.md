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

- `eph_00`–`eph_09` · `kin_00`–`kin_07` · `fip_00_explore`
  - `eph_07_bout_encoding`: within-trial (go-responsive) vs ITI LC-unit encoding of
    tongue-movement bouts, ported from `tongue_kinematics_ephys_intertrialmovs.ipynb`.
    Movement-bout-derived (`annotate_movement_bouts` + `classify_bout_times` /
    `get_session_bout_times`, now in `ephys_utils.py`) is primary; lick-bout-derived
    (`licks["bout_start"]`) is a lighter-weight robustness check (§9). Population PETH
    (heatmap, mean±SEM overlay, waterfall, go-cue reference) reuses `eph_00`'s
    `make_rp_and_events`/`compute_psth` raster path; per-unit go-responsive-vs-ITI Δhz
    is registered through `encoding_methods`/`per_unit_stats_registry` so it composes
    with `eph_01`–`eph_04`. Only the bout-helper verification (§4) has actually run,
    locally against the pooled parquet — everything needing spike times is Code Ocean
    only and unexecuted. `tongue_kinematics_ephys_intertrialmovs.ipynb` now has no
    remaining port gate. See `TODO.md`.
  - `kin_02_latency`: §8–§10 carry the RT + IMI decomposition ported from
    `tongue_latency` (Δt de-shift, KS collapse test, noise propagation w/ bootstrap
    CI, cross-session grand mean); §11 carries that notebook's single-session
    illustration (example trial, rasters, colored histograms) — reassigned here
    from the `kin_05` plan since it was already ported. `tongue_latency.ipynb` has
    no remaining port gate. See `TODO.md`.
  - `kin_05_nonlick_movements`: lick↔movement correspondence tally (§3), non-lick
    kinematic profile extending `kin_01` §5 (§6), per-trial lick/non-lick structure
    (§7), and pre-cue-response preparatory-movement timing (§8) ported from
    `tongue_kinematics` + `cueresponse` cell 43. Definitional, not a QC filter — see
    the noise-definition memory. §3's "licks without movement" tally and §4/§5's
    example figures are Code-Ocean-only (need per-session `nwb_df_licks.parquet`);
    everything else runs on the pooled parquet. `tongue_kinematics.ipynb` now gates
    on `kin_07` alone. See `TODO.md`.
  - `kin_06_lick_geometry_choice`: does the direction of a preparatory tongue movement
    predict which spout the animal licks? Landmark frame (§3) and cue-response endpoints
    over it (§4) are Code-Ocean-only — they need per-session **spout** keypoint means
    (`kps_raw_*.parquet`). Everything else pools across 44 sessions, because the **jaw**
    origin comes, per session, from `get_jaw_positions()` (§2.1), which **prefers the real
    jaw keypoint** (`kps_raw_jaw.parquet`, Code-Ocean-only) and falls back to an algebraic
    reconstruction — `endpoint_x/y` and `max_*_from_jaw` are absolute pixel positions (not
    jaw-relative, despite the names; the distances are `max_x_distance` / `max_y_distance`)
    — only where that file is missing, which locally is all 44 sessions (checked: no other
    pooled column, e.g. `startpoint_x/y`, sits at a fixed point either — all noisier by
    10-40x than the fallback's own internal residual). `get_jaw_positions()` reports the
    keypoint-vs-algebraic gap automatically wherever both exist, replacing the old
    single-session cross-check in §3. Lick-vs-non-lick excursion
    geometry (§5, the `tongue_kinematics` 60 == `cueresponse` 95 duplicate, ported once),
    non-lick endpoints by ordinal (§6), P(right lick) vs pre-lick geometry (§7),
    ridge-logistic lick-side decode (§8) and pre-lick → cue-response displacement (§9).
    The decode uses `GroupKFold` on session (held-out-session AUC 0.835 ± 0.100) plus
    per-session fits (median 0.943) and a within-session shuffled-label null (0.530) —
    the source's single-session `train_test_split` would leak session identity once
    pooled. §8.4 quantifies the block-structure confound rather than only noting it
    (P(stay) = 0.910; previous choice alone AUC 0.907; kinematics 0.77–0.82 within
    previous-choice strata). `plot_standard_lick_landmarks` lives in the notebook, not
    `plotstyle.py` — one consumer. Executed end-to-end locally except §3–§4.
    `tongue_kinematics_cueresponse.ipynb` now gates on `kin_07` alone. See `TODO.md`.
  - `kin_07_value_encoding`: do behavioural-model latents (Q, RPE) explain tongue
    kinematics? The one topic the refactored series did not track at all — merged from
    the two partly-overlapping single-session copies in `tongue_kinematics` (111–138) and
    `cueresponse` (17–28). Latents come from `get_mle_model_fitting`; the notebook
    imports it from `aind_analysis_arch_result_access.df_mle_model_fitting` (the
    `han_pipeline` path both sources use is now a deprecated shim) with a fallback.
    **Pooled, not single-session**: the fetch is ~1 s/session, 40 of 44 sessions have a
    `QLearning_L2F1_CKfull_softmax` fit (subject 751004's four sessions have no MLE
    records), and pooling is what makes **session** available as the sampling unit.
    Three statistical problems in the sources are corrected rather than inherited —
    pseudo-replication (trial-level latents merged onto 246k movement rows; fixed by
    aggregating to one row per trial *and* testing across sessions), no
    multiple-comparison control (§6's 24×11 grid is BH-FDR corrected throughout), and
    over-read correlated features (§6.1 measures the redundancy: five kinematic pairs
    exceed ρ = 0.99, three columns are literally the same variable once within-session
    z-scored, so only the `GroupKFold`-on-session RidgeCV R² panel is a genuine
    predictive measure). `TODO.md`'s claim that `attach_model_latents_to_trials` is
    defined identically in the two sources is **wrong** — they differ in the sign of
    `q_diff`; ported once as `R − L`, matching `kin_06`'s frame. **Code Ocean only and
    effectively unexecuted**: §6.1 is the only section with real output; the §6 screen
    machinery was verified against a synthetic latent instead. Both
    `tongue_kinematics.ipynb` and `tongue_kinematics_cueresponse.ipynb` are now fully
    replicated, but **their archive gates stay open until this runs on Code Ocean**.
    See `TODO.md`.
  - `eph_09_structural_axes`: does the RT-encoding spatial gradient align with LC's
    structural organization? Fits the RT-encoding axis (`T_rt`) and its baseline control
    (`T_rt_bl`) — the step `eph_08` skipped — plus three structural axes: waveform
    features (CCA), MERFISH transcriptomics (CCA), retrograde tracing (LDA). Pairwise
    direction comparison (angle, Wald W, χ² p, bootstrap p), three-plane arrow-and-cone
    figure, azimuth-elevation bootstrap scatter, and projection scatters onto each
    structural axis. All axis math is `spatial_axes.py`; RT stats go through
    `AnalysisSpec`/`fit_encoding`/`per_unit_stats_registry` (as `eph_01`) and unit QC
    through `data_loading`, so no inline copies were added. Each structural axis sits
    behind `HAS_WAVEFORM`/`HAS_MERFISH`/`HAS_RETRO`, so a missing asset drops that axis
    rather than failing. **Code Ocean only and entirely unexecuted** — written 2026-09-15,
    skip path verified locally only. MERFISH additionally needs the `scanpy` Dockerfile
    layer added the same day, i.e. an image rebuild. `spatial_axis_comparison_rt_encoding_
    update.ipynb` is fully replicated but its archive gate stays open until this runs.
    See `TODO.md`.
- Modules: `data_loading.py`, `ephys_utils.py`, `encoding_methods.py`, `encoding_plots.py`,
  `per_unit_stats_registry.py`, `spatial_encoding.py`, `spatial_axes.py`, `plotstyle.py`,
  `ccf_utils.py`
  - `per_unit_stats_registry.py`: distinct results-store + FDR + cross-analysis `compare`
    (used by `eph_01/02/03/04/06`); not superseded by `ephys_utils`/`encoding_methods`.
  - `ccf_utils.py`: imported by `eph_08`, `eph_09` (`pir_to_lps`, `project_to_plane`,
    `ccf_pts_convert_to_mm`) + `spatial_encoding.py`. **Not** imported by
    `spatial_axes.py` — that module is coordinate-frame agnostic and takes coords as
    plain Nx3 arrays, so the CCF conversion stays in the notebooks.
  - `spatial_encoding.py`: `SpatialEncoder` — CCF maps, subgroup maps, spatial-dependence
    permutation tests. Contains **no axis-fitting machinery**; that is `spatial_axes.py`'s
    job (landed 2026-09-15).
  - `spatial_axes.py`: fitting and comparing 3-D gradient *directions* —
    `fit_spatial_axis_linear`/`_cca`/`_LDA` + bootstrap wrappers,
    `compare_bootstrap_directions`, `cone_half_angle`, `vectors_to_az_el`, and the two
    plot helpers. Consumers: `eph_08` (waveform axis) and `eph_09`. Deliberately separate
    from `spatial_encoding.py`: *where* a statistic is large vs. *in which direction* it
    changes are different questions with different inputs.
  - `ephys_utils.py`: spike counting, session bundles, `all_counts_df` — **and** behavior-derived
    per-trial features (`build_trial_features`) that exist to serve ephys alignment. Bout
    segmentation (`annotate_movement_bouts`, `classify_bout_times`, `get_session_bout_times`)
    lands here for the same reason — landed 2026-09-15 with `eph_07`, see `TODO.md`.
- The split between these modules and the library's `tongue_kinematics_utils` / `tongue_ephys`
  is currently ad hoc — both sides carry "kinematics utils" and "ephys utils", and the library
  also ships plot functions that ignore `plotstyle.py`. Defining that boundary is its own
  `TODO.md` item; read it before adding a module or promoting anything to the library.

### Planned additions (from the HOLD ports — see `TODO.md`)

**Nothing remains planned.** `kin_07_value_encoding.ipynb` — the last of the six ports —
landed 2026-09-15 and is in KEEP above. What remains is not writing but *running*: `kin_07`
and `eph_09` are both written and unexecuted on Code Ocean, and those two runs are the only
things holding open the archive gates for three of the five HOLD notebooks.

**No new modules remain planned.** `spatial_axes.py` landed 2026-09-15 with
`eph_09_structural_axes.ipynb` (both now in KEEP above), and `eph_08` was refactored onto it,
deleting its two private inline copies of the CCA fit/bootstrap. `eph_07_bout_encoding.ipynb`
is done too — its orphan `annotate_movement_bouts` and the within-trial/ITI classifier folded
into the existing **`ephys_utils.py`**, matching the `build_trial_features` precedent there
(behavior-derived features that serve ephys alignment); a module with one consumer isn't worth
the file. Rationale in `TODO.md`'s `eph_07` item.

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
- `tongue_lickometer.ipynb` — can Lightning-Pose detect a lick *defined as tongue–spout
  contact*? The repo's only measurement of detection **precision**; the library's
  `coverage_pct` path is contact-agnostic and recall-only. Does not currently run (every
  domain import is a bare name for a module now in the library). Being repaired and refocused
  as a two-implementation comparison — the current threshold+refractory detector vs a
  gap-aware hysteretic one — on the single session it already uses. Target name
  `val_02_spout_contact_detection.ipynb`; the rename is **not** a prerequisite. See `TODO.md`.

## HOLD — port unreplicated content before archiving (see `TODO.md`)

Audited cell-by-cell on 2026-09-11. The unreplicated content does **not** map one-notebook-to-one-port:
it clusters into five questions that cut across these files, with duplication between them. Read
the "Port plan for the five HOLD notebooks" overview at the top of `TODO.md` before starting any
of them.

| File | Unreplicated content | Ports into |
|---|---|---|
| `tongue_latency.ipynb` | ~~RT + IMI decomposition (Δt de-shift, KS collapse, noise propagation w/ bootstrap CI); single-trial example fig; trial rasters by movement type / ordinal~~ **all done, in `kin_02` §8–§11** | `kin_02` §8–§11 (done) — **no remaining gate; ready to archive** |
| `tongue_kinematics_ephys_intertrialmovs.ipynb` | ~~within-trial vs ITI bout-aligned ephys encoding; sole copy of `annotate_movement_bouts`~~ **done, in `eph_07`** | `eph_07` (done) — **no remaining gate; ready to archive** |
| `spatial_axis_comparison_rt_encoding_update.ipynb` | ~~RT-encoding spatial axis fit (`eph_08` skipped it), MERFISH (CCA) + retrograde (LDA) axes, bootstrap direction comparison, confidence cones~~ **all written, in `eph_09` + `spatial_axes.py`** | `eph_09` (done, **unrun**) — gate open until `eph_09` executes on Code Ocean. `eph_08` cleared its own check 2026-09-16: reproduces the poster figure at r=0.184, p=0.0681, n=99, after fixing the spike-count windows and the ML fold (see `TODO.md`). |
| `tongue_kinematics.ipynb` | ~~lick↔movement correspondence (licks w/o movements, movements w/o licks, multi-lick); per-trial non-lick structure~~ **done, in `kin_05` §3, §5–§8**; ~~kinematics vs behavioural-model latents (Spearman/MI/RidgeCV/RF, prev-trial RPE)~~ **all written, in `kin_07` §3–§7** | `kin_05` (done) **+** `kin_07` (written, **unrun**) — gate open until `kin_07` executes on Code Ocean |
| `tongue_kinematics_cueresponse.ipynb` | ~~jaw/spout landmark geometry + endpoints by event; choice prediction from pre-lick kinematics (ridge-logistic, AUC, binned P(right lick))~~ **done, in `kin_06` §3–§9**; ~~Q-value encoding~~ **done, in `kin_07` §3–§6** | ~~`kin_06`~~ (done) **+** `kin_07` (written, **unrun**) — gate open until `kin_07` executes on Code Ocean |

Two corrections to the earlier reading of this table:

- **`cueresponse` is not just "spatial geometry."** The landmark figures are the setup for a
  choice-prediction analysis (cells 43–56) that is the actual result.
- **Kinematics × behavioural-model latents is an untracked topic**, present in *two* of these
  notebooks and in no `kin_*`/`eph_*` notebook. It gates two archivals, so neither
  `tongue_kinematics` nor `tongue_kinematics_cueresponse` can be retired without `kin_07`.
  `kin_07` was written 2026-09-15 and covers both; because it is Code Ocean-only and
  unexecuted, both gates stay open until it runs.

Already covered elsewhere — do **not** port: `tongue_kinematics` cells 78–79 (lick-detection
FP/FN parameter sweep) duplicate `tongue_lickometer.ipynb`, which is KEEP.

Also tracked in `TODO.md`: consolidate the duplicated `compute_outbound_metrics`
(`add_outbound` + `tongue_movements_all`) into the library. `annotate_movement_bouts` is the
same orphan-code pattern — it lands in `ephys_utils.py` with `eph_07`, and becomes a library
candidate once `eph_07` has exercised it and the library/repo boundary criteria exist.

## ARCHIVE — ready to move to `code/archive/`

Superseded by the new series / modules / library:
- `rt_ols_registry_pipeline.ipynb`, `registry_usage_example.py` → `ephys_utils` + `encoding_methods` + registry + `eph_01`
- `umap.ipynb` → `kin_03`
- `tongue_kinematics_ephys.ipynb`, `tongue_kinematics_ephys_figures.ipynb` → `eph_*` / `kin_04`
- `spatial_axis_comparison_rt_encoding.ipynb` — **not a superseded duplicate; keep its
  outputs.** It is the notebook that generated the poster figure
  `rt_response_projection_abs.svg` (cell 42, `execution_count` 37, 2026-05-04, committed
  in `2f20188`), and its **stored outputs are the only surviving record** of the
  spike-count windows, waveform axis and unit counts behind that figure — they are what
  made the 2026-09-16 replication possible. `_update` was created afterwards, with its
  outputs cleared, and could prove none of it. Archived is the right place for it; do not
  strip its outputs.
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

1. `kin_02` §8–§10 — pooled parquet only, no new dependencies, fully local-testable. — **done**
2. `kin_05_nonlick_movements` — mostly pooled; two partial Code Ocean-only sections
   (§3's licks-without-movement tally, §4's/§5's example figures). — **done**
3. `eph_07_bout_encoding` (+ bout helpers into `ephys_utils.py`) — reuses `eph_00`'s
   raster/PETH helpers. — **done**
4. `spatial_axes.py` + `eph_09_structural_axes`, then refactor `eph_08` onto the module. —
   **written 2026-09-15; assets confirmed mounted, `scanpy` added to the Dockerfile.**
   Not yet run on Code Ocean, so `eph_08`'s poster-figure output is unconfirmed and the
   source notebook's archive gate stays open. See `TODO.md`.
5. `kin_06_lick_geometry_choice` — pooled in a reconstructed jaw-centered frame; two
   Code Ocean-only sections (§3–§4, per-session spout keypoints). — **done**
6. `kin_07_value_encoding` — pooled over the 40 sessions with an MLE fit; new dependency
   on `get_mle_model_fitting`, confirmed live. — **written 2026-09-15, unrun on Code Ocean.**

Archive a HOLD notebook only when **every** gate in the HOLD table above is met. All six ports
are now *written*; three of the five HOLD notebooks wait only on `kin_07` and `eph_09`
executing on Code Ocean. Two — `tongue_latency.ipynb` and
`tongue_kinematics_ephys_intertrialmovs.ipynb` — have no remaining gate at all.
