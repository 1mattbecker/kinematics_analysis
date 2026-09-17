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
