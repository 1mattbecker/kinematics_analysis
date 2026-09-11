# TODO — kinematics_analysis

Deferred work items. Newest first. Dates are YYYY-MM-DD.

---

## Port the ITI vs task bout-aligned ephys analysis into a new `eph_07`

_Logged 2026-09-11._

### Why

`code/tongue_kinematics_ephys_intertrialmovs.ipynb` (43 cells) is a distinct ephys analysis with
no equivalent in the new `eph_*` series: it groups movements into **bouts**
(`annotate_movement_bouts`, gap threshold) and compares LC unit responses to **within-trial
(go-response) bouts vs inter-trial-interval (ITI) bouts** — bout-aligned rasters/PETHs, population
overlays/waterfalls, and a per-unit `go_responsive` vs ITI Δhz encoding scatter. This is the ephys
counterpart to the "covert / spontaneous preparatory movement" story in `tongue_latency.ipynb`.

The notebook already imports the refactored modules (`data_loading`, `ephys_utils`), so it is
partway to the new structure.

### What to do

- Create `eph_07_bout_encoding.ipynb`: bout segmentation → within-trial vs ITI bout event times →
  reuse `eph_00`'s raster/PETH helpers and `encoding_methods` for the per-unit go-responsive vs ITI
  comparison.
- Pull bout event times from `all_tongue_movements` (has `start_time`, `trial`); align to spikes via
  the cached `filtered_ephys.pkl`.
- After it lands, archive `tongue_kinematics_ephys_intertrialmovs.ipynb` (see REORG.md).

---

## Port the MERFISH + retrograde-tracing axis comparisons into `eph_08`/`eph_09`

_Logged 2026-09-11._

### Why

`code/spatial_axis_comparison_rt_encoding_update.ipynb` (41 cells) compares the RT-encoding spatial
axis against **four** structural axes — waveform (CCA), **MERFISH transcriptomics (CCA)**,
**retrograde tracing (LDA)** — with bootstrap direction comparison, 95% **confidence cones**
(azimuth-elevation), and projection scatters. The refactored `eph_08_waveform_axis.ipynb` ported
**only the waveform CCA axis** (§4–5). MERFISH, retrograde, LDA, and the multi-axis cone
visualization are not replicated anywhere in `eph_04`/`eph_08`.

(`spatial_axis_comparison_rt_encoding.ipynb` without `_update` is an older duplicate — archive it;
keep `_update` as the source for this port.)

### What to do

- Extend `eph_08` (or add `eph_09_structural_axes.ipynb`) with the MERFISH (CCA) and retrograde
  (LDA) axis fits, the bootstrap `compare_bootstrap_directions` machinery, and the confidence-cone /
  projection-scatter figures from `_update`.
- The comparison axes load "from Han's capsule" — confirm those inputs are still reachable on Code
  Ocean before porting.
- After it lands, archive both `spatial_axis_comparison_rt_encoding*.ipynb` (see REORG.md).

---

## Port the RT+IMI latency story out of `tongue_latency.ipynb` before archiving it

_Logged 2026-09-11._

### Why

`code/tongue_latency.ipynb` is a legacy notebook mostly superseded by `kin_02_latency.ipynb`,
but `kin_02` only absorbed the clean parts (latency distributions by ordinal, k=1 log-normality,
cross-ordinal correlation). Three pieces of the notebook's scientific story were **not** ported and
exist nowhere in the new `kin_*`/`eph_*` series:

1. **RT + IMI decomposition** — reaction time modeled as a sequence of inter-movement intervals
   conditioned on movement number (uses `bootstrap_std_ci`). This is the core "nice story."
2. **Single-trial example figure** — per-trial illustration of movements/licks colored by type
   (`trial_to_plot`, `coerce_bool`, `color_for_row`); good explanatory/illustrative figure.
3. **"Covert preparatory movements" narrative** — the interpretation tying later-ordinal latency to
   additional covert preparatory tongue movements. Links to
   `tongue_kinematics_ephys_intertrialmovs.ipynb` (inter-trial movements), which is also
   not yet replicated.

### What to do

- Port (1)–(3) into `kin_02_latency.ipynb`, or a new `kin_05_latency_decomposition.ipynb` if it's
  cleaner to keep the ordinal-latency material separate from the RT-decomposition story.
- Reuse the `all_tongue_movements.parquet` load path already used by `kin_02` (no spike data needed).
- Only after the port lands: archive `tongue_latency.ipynb` (see REORG.md).

---

## Port the lick↔movement correspondence analysis out of `tongue_kinematics.ipynb`

_Logged 2026-09-11._

### Why

`code/tongue_kinematics.ipynb` (139-cell original monolith) is mostly superseded — its
processing/QC (`segment_movements`, tracking QC, refractory filtering) now lives in the library
(`tongue_kinematics_utils`). But its **lick↔movement correspondence** analysis has no equivalent in
the new `kin_*` series: licks without movements, movements without licks, movements with multiple
licks, and the associated duration / max-excursion comparisons across those categories.

### What to do

- Port that section into `kin_00_movement_qc.ipynb` or `kin_01_population.ipynb` (whichever frames
  it best), using the `all_tongue_movements.parquet` + lick annotations already loaded there.
- After it lands, archive `tongue_kinematics.ipynb` (see REORG.md).

---

## Port the cue-response spatial geometry out of `tongue_kinematics_cueresponse.ipynb`

_Logged 2026-09-11._

### Why

`code/tongue_kinematics_cueresponse.ipynb` (115 cells) has cue-response **spatial geometry** not
replicated in `kin_*`: jaw↔spout landmark positions, tongue endpoints colored by event type, and the
spout-relative excursion geometry. `kin_*` currently carries endpoint / excursion_angle / trajectory
columns but not these landmark-referenced spatial figures.

### What to do

- Port the landmark geometry + endpoints-by-event figures into a `kin_*` notebook (new
  `kin_06_cue_response_geometry.ipynb` if it doesn't fit cleanly into an existing one).
- After it lands, archive `tongue_kinematics_cueresponse.ipynb` (see REORG.md).

---

## Fold outbound metrics into the library's movement aggregation step

_Logged 2026-09-11._

### Why

The outbound-phase metrics (`out_duration`, `out_peak_velocity`, `out_mean_velocity`,
`out_total_distance` — computed over the outbound segment only, start → max-jaw-excursion frame)
are **not** part of the standard segmentation/aggregation in
`aind-dynamic-foraging-behavior-video-analysis`. The library's
`kinematics/tongue_kinematics_utils.py::aggregate_tongue_movements()` only emits *whole-movement*
metrics (`peak_velocity`, `duration`, `total_distance`, excursion endpoints). No branch
(`main`, `LC_manuscript`, `video_alignment`) computes an outbound split.

As a result the logic lives as orphan code in this repo, duplicated in two places:
- `code/add_outbound.ipynb` — the canonical generator (`compute_outbound_metrics` +
  `augment_movs_with_outbound_in_place`), which backfills `out_*` into each per-session
  `tongue_movs.parquet` after the fact.
- `code/tongue_movements_all.ipynb` cell 30 — a duplicate copy of `compute_outbound_metrics`.

Every downstream consumer already depends on these columns: `ephys_utils.py` (`kcols` →
`first_move_out_*` / `cue_response_out_*` in `all_counts_df`), `eph_02/03/04`, and `kin_00/01/03/04`.

### What to do

- Move `compute_outbound_metrics()` into the library, computing `out_*` **inside**
  `aggregate_tongue_movements()` (it already has `movement_id`, `x`, `y`, `v`, `time_in_session`
  and the jaw keypoint it needs) so every session's movement table carries `out_*` by default —
  no separate augmentation pass.
- Once the library emits `out_*` natively: delete the two duplicate defs
  (`add_outbound.ipynb`, `tongue_movements_all.ipynb` cell 30), retire the
  `augment_movs_with_outbound_in_place` backfill, and re-run the per-session pipeline so the
  intermediates are regenerated with `out_*` baked in from segmentation.
- Verify column parity against the current `all_tongue_movements_04022026.parquet` before/after
  (same values, same dtypes) so `ephys_utils.py` and the `kin_*`/`eph_*` notebooks are unaffected.

### Notes

- Do this in the library repo (its own branch/PR), then bump the pin here.
- Unblocks archiving `add_outbound.ipynb` and simplifies `tongue_movements_all.ipynb`
  (see REORG.md).

---

## Upgrade the Code Ocean environment off Python 3.9

_Logged 2026-09-07. Not scheduled — do this deliberately, on a branch, not mid-analysis._

### Why

`environment/Dockerfile` builds from
`codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04`. Python 3.9 reached
end-of-life in October 2025. That single pin is the root cause of several unrelated-looking
workarounds scattered through this repo:

1. **`rachel-analysis-utils` needs `--ignore-requires-python`.** The package declares
   `requires-python>=3.10`. The Dockerfile installs it in an isolated `RUN` layer with
   `--ignore-requires-python` specifically so that flag does not leak into the main pip block
   (where it would let pip resolve a too-new numpy and break the build). See the comment above
   that layer.
2. **`rachel_analysis_utils.analysis_utils` cannot be imported at all** on 3.9 — it has a
   nested-quote f-string at `analysis_utils.py:294` (`f'{x.split('_dff')...}'`), which is 3.12+
   syntax. Because of this, `fip_00_explore.ipynb` reimplements `enrich_df_trials`'s
   `num_reward_past` locally as `enrich_streaks` (unit-tested to match the package). That
   reimplementation exists *only* to dodge the import and can be deleted after the upgrade.
3. **The repo-wide "Python 3.9 compatible syntax only" rule** in `CLAUDE.md` — no `|` type
   unions, no structural pattern matching, no walrus in complex contexts — exists solely to
   match this image.

Upgrading the base image removes all three.

### Risk (why this is not a quick change)

The main pip block pins a mutually-consistent stack that was resolved against 3.9:

```
spikeinterface[full]==0.100.0   scipy==1.13.0        pynwb==3.0.0
hdmf-zarr==0.11.0               zarr==2.18.2         statsmodels==0.14.2
pyarrow==21.0.0                 seaborn==0.13.2      opencv_python==4.11.0.86
moviepy==1.0.3                  scikit-image==0.24.0 aind-ephys-utils==0.0.15
```

Several of these will need version bumps on a newer interpreter, and `spikeinterface[full]`
in particular pulls a large transitive tree. ~41 notebooks depend on this environment, with
the `eph_*` and `kin_*` families being the most exposed (they use spikeinterface, pynwb and
the ephys stack directly). A bad upgrade silently breaks working analyses.

### Suggested approach

- Do it on a dedicated branch, and use Code Ocean's environment versioning so the current
  working environment stays recoverable.
- Bump the base image to the Code Ocean `python3.11` JupyterLab image (matches
  `aind-motion-energy`, which already targets 3.11).
- Relax the pins in the main block to `>=` where possible, rebuild, then re-pin to whatever
  actually resolves — do not hand-guess versions.
- Drop `--ignore-requires-python` from the `rachel-analysis-utils` layer and fold it back
  into the main pip block.
- Verification pass, in rough order of blast radius: `eph_00`, `kin_00`, then `fip_00`.
  Confirm `import rachel_analysis_utils.analysis_utils` succeeds, then delete the local
  `enrich_streaks` reimplementation in `fip_00_explore.ipynb` and switch back to
  `enrich_df_trials`.
- Remove the Python 3.9 syntax rule from `CLAUDE.md` once the image is live.

### Related cleanups unblocked by this

- `fip_00_explore.ipynb` cell 3 pip-installs `fastparquet` at runtime because the image does
  not ship it (`rachel_analysis_utils.load_nwb_list` hardcodes `engine="fastparquet"`).
  That belongs in the Dockerfile pip block regardless of the Python version — it does not
  need to wait for this upgrade.
