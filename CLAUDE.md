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
├── code/               # All analysis code — notebooks and scripts
├── environment/        # Docker + postinstall scripts (DO NOT MODIFY)
├── metadata/           # Project metadata
├── .codeocean/         # Code Ocean config (DO NOT MODIFY)
└── CLAUDE.md           # This file
```

---

## Key notebooks and scripts

*(Update this section as files are added or renamed)*

- Tongue kinematics pipeline: lick detection, bout classification, outbound phase metrics
- `compute_outbound_metrics()` — core function for kinematic feature extraction
- UMAP embedding of movement bouts
- Partial correlation analysis
- `fip_00_explore.ipynb` — fiber photometry access/plotting. Structured as imports →
  data loading → data processing → data viz. Loads the saved parquet hierarchy via
  `rachel_analysis_utils.nwb_utils.load_nwb_list`; `USE_CURATION` + `CURATION_FILE`
  (`DA_NE_4channel_datacuration_firstpass`, which carries `correct_mapping`) annotate
  `df_fip['intended_measurement']`. Processing picks three example signals (NAc DA/dLight,
  PL/GCaMP, NAc ACh/rAch) and aligns motion energy to the FIP clock; `pearsonR` series are
  excluded (signal-signal correlations, not photometry). Viz: full/60s traces, peri-go-cue
  averages, FIP↔motion-energy onset alignment, and ME×FIP cross-correlation. Runs on Code
  Ocean (data asset `6babbf3d…`). First step toward correlating FIP with tongue kinematics.

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
