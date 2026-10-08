# `code/archive/` — superseded and scratch notebooks

Kept for provenance, **not** for running. Nothing in `code/` imports anything here (verified
before archiving: every apparent hit was the library module `tongue_kinematics_utils`, a
substring match, or a provenance line in a comment). Notebooks were moved with `git mv`, so
history is intact, and `main` is frozen and mirrored in
`AllenNeuralDynamics/LC-ephys-tonguemovements` — nothing here is at risk.

This file replaces `REORG.md`, retired 2026-09-16 once the reorg completed. The current map of
`code/` lives in `CLAUDE.md`; actionable work lives in `TODO.md`. To read the old plan:
`git show 52585bf:REORG.md`.

---

## Do not strip outputs from `spatial_axis_comparison_rt_encoding.ipynb`

This is the one archived notebook whose **stored outputs matter**. It generated the poster
figure `rt_response_projection_abs.svg` (cell 42, `execution_count` 37, 2026-05-04, committed
in `2f20188`), and its outputs are the only surviving record of the spike-count windows,
waveform axis and unit counts behind that figure. They are what made the 2026-09-16
replication possible — `_update` was created afterwards with its outputs cleared and could
prove none of it.

The two things that replication turned up, both now fixed in `eph_08`/`eph_09`:
`count_window_s=(0.0, 0.5)` / `baseline_window_s=(-2.0, 0.0)` (not the 0.2 s / 1 s pair that
had been inherited from a commented-out config), and the ML fold onto the **positive** side.
Full account in `TODO.md`.

---

## What was archived and why

### FIP notebooks, reorganized by analysis type (archived 2026-10-08)

The `fip_*` series was regrouped into `fip_10`–`fip_17`, one notebook per kind of analysis, all on
the CSV-curated assets. Cells were copied, not rewritten; the table says where each part went.
Everything shown in the 2026-10-07 talk ("DA, NE and movement") is in the new notebooks.

| Notebook | Where its content went |
|---|---|
| `fip_00_explore.ipynb` | Retired. Single-session exploration on the JSON-curated `DA_NE_4channels` asset; superseded by `fip_10`/`fip_12` and `men_00` |
| `fip_01_movement_value_coding.ipynb` | Retired. ME by RPE bin / reward streak on the old asset; superseded by `fip_16` §6 |
| `fip_02_ne_only_events.ipynb` | Retired. NE-only onsets in one session; superseded by `fip_11` |
| `fip_03_da_ne_commonality.ipynb` | Retired. ME from DA/NE variance partition, one animal; superseded by `fip_13` §1 |
| `fip_04_da_ne_xcorr.ipynb` | Retired. DA × NE xcorr on the old asset; superseded by `fip_10` §1 (CSV-curated rebuild) |
| `fip_05_da_ne_rpe_coupling.ipynb` | §1, §2, §4a–c → `fip_16`; §3d–e and §5's threshold sweep → `fip_11` §2–3; §3f and §4d–e → `fip_13` §5. Dropped: the example-trace figure, §3a–c (go-cue latency; `fip_15` §2 has it on Rachel's pipeline) and the rest of §5 (same as `fip_11` §1) |
| `fip_07_da_ne_summary.ipynb` | Fig 1 → `fip_10` §1; Fig 2 → `fip_11` §1; Fig 3 → `fip_15` |
| `fip_08_movement_da_ne.ipynb` | §0b → `fip_10` §2; §1 a–b and §5 → `fip_12`; §1 c–d → `fip_10` §2; §1 e–f, model, §2, kernels → `fip_13`; §3 → `fip_16` §6; §4 → `fip_11` §4 |

`fip_06_rpe_split` and `fip_09_task_me_models` were renamed (not archived) to `fip_17_rpe_split`
and `fip_14_task_model_variants`.

### Ported into the `kin_*`/`eph_*` series (the former HOLD set, archived 2026-09-16)

| Notebook | Ported into |
|---|---|
| `tongue_latency.ipynb` | `kin_02` §8–§11 + `kin_05` |
| `tongue_kinematics_ephys_intertrialmovs.ipynb` | `eph_07` + bout helpers in `ephys_utils.py` |
| `spatial_axis_comparison_rt_encoding_update.ipynb` | `eph_09` + `spatial_axes.py` (and `eph_08` §5) |
| `tongue_kinematics.ipynb` | `kin_05` + `kin_07` |
| `tongue_kinematics_cueresponse.ipynb` | `kin_06` + `kin_07` |

### Superseded by the new series, modules, or the library (archived 2026-09-15)

- `rt_ols_registry_pipeline.ipynb`, `registry_usage_example.py` → `ephys_utils` +
  `encoding_methods` + `per_unit_stats_registry` + `eph_01`
- `umap.ipynb` → `kin_03`
- `tongue_kinematics_ephys.ipynb`, `tongue_kinematics_ephys_figures.ipynb` → `eph_*` / `kin_04`
- `spatial_axis_comparison_rt_encoding.ipynb` → `eph_08` / `eph_09` (**but see above**)
- `old_functions_tongue_kinematics.ipynb`, `cue_response_lick_processing_example.ipynb`,
  `tongue_segmentation_test.ipynb`, `batch_clips.ipynb`, `create_labeled_clip.ipynb` —
  functions promoted to, or already owned by, the library
- `example.ipynb`, `extract_tongue_kinematics.ipynb` — scratch
- `event_timeline.ipynb` — task event-timeline plot

### `reference/`

- `F_ephys_behavior_action&outcome.ipynb` — a collaborator's outcome/action auROC analysis,
  kept as reference rather than as this repo's code.
