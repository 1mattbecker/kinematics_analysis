# Changelog

Notable changes to this project. Newest first. Dates are YYYY-MM-DD.

## 2026-09-17

### `val_04_lickometer_qc`: the short answer to dynamic-foraging-processing#96

A new notebook carrying the method, the per-session miss rate across 53 sessions, the limitations,
and two QC metrics. 21 cells against `val_03`'s 35. The threshold sweep table, the single-event
trajectory panels, the video-clip extraction, the per-trial flag and the high-confidence event list
stay in `val_03`, which remains the workup.

- Everything runs at `D0` = 10 px, 2.7x the 3.67 px `tongue_tip_center` error. The sweep survives as
  the calibration curve in §2: recall 0.45 / miss 0.9% at 10 px, recall 0.90 / miss 3.6% at 30 px.
- `val_03` §8 listed five QC metrics, A-E. This keeps two. The session miss rate, gated on
  `n_pose` >= 300 and `lick_coverage` >= 0.90 and flagged on a Wilson lower bound above 10%, ranks
  sessions for review. The within-subject day-to-day range, flagged above 10 points, separates a
  failing rig from a quiet animal and recovers `763590`, which the first one's `n_pose` gate drops.
- `791691` runs 38.2 / 1.9 / 31.1 / 13.0% on four consecutive days. Median session is 2.8%, range
  0.0-38.2%.
- §5 lists six limitations, each with a number attached: no ground truth, since precision is
  measured against the instrument under test; `D0` biases in both directions; pixel scale varies
  1.8x across sessions; `spearman(miss_rate, tracked_frac)` = +0.277; one camera against
  session-mean spouts; #96's own session has no pose output.
- §6 writes `val_04_lickometer_qc.csv`, one row per session with both flags.

Unexecuted: needs `session_analysis_mlk/*/intermediate_data/` on Code Ocean. Numbers quoted in the
prose come from the 53-session run recorded in `val_03`.

## 2026-09-16

### `val_03_missed_licks`: §7 — `lick_coverage`, Wilson intervals, within-subject spread

The executed run gave `spearman(miss_rate@10px, tracked_frac) = +0.277`, against +0.329 at 30 px.
Conditioning on a tight `d` did **not** remove the correlation.

`tracked_frac` is the wrong covariate. It counts every tongue-visible frame, so it rises with
non-lick tongue movement as well as with tracking quality; a session with more tongue-out time
produces more excursions that pass within 10 px of a spout without contacting. Meanwhile
`spearman(miss_rate, n_pose) = -0.081` — the number of qualifying events does not predict the
rate, so the correlation is not a sample-size effect.

- **`lick_coverage`** replaces it: of the lickometer's licks, the fraction with a tracked tongue
  within ±100 ms. It conditions on a lick having happened, so behaviour divides out and what is
  left is pose sensitivity. Both are printed with their ranges; the correlation is reported as
  "n/a (no spread)" when a covariate is near-constant, which is itself the answer — the example
  session had coverage 99.98%, and if that holds everywhere then pose sensitivity is uniform and
  the `tracked_frac` correlation is behavioural.
- **Wilson 95% intervals** on every miss rate, drawn on the ranking. `n_pose` spans 145 to 3928,
  so intervals run from 0.9 to 15.1 points wide. `MIN_N_POSE = 300` marks sessions as reported but
  not ranked — under the old ranking, positions 2 and 4 were the two smallest-n sessions in the
  set (`763590`: 48/145 and 43/214).
- **Within-subject spread** as a second table. `791691` ran 1.8% / 13.0% / 31.1% / 38.2% on four
  consecutive days; `782394` ran 0.0-1.6%. A 20x swing inside one animal in one week is not
  behaviour, and is a stronger flag than a high rank.

§3 and §7 figures moved to `layout="constrained"`.

### `val_03_missed_licks`: direct prose throughout; §4 and §5 conclusions corrected

Markdown 13,036 -> 7,986 chars. Cut the contrastive framing and the reader-instruction asides:
"triage, not prevalence", "characterizations, not recommendations", "Panel 2 is the deliverable",
"the honest QC output is", "The central caveat, stated once up front", and all three
"What to take from it" headers. Sections named after what they compute.

**Two §4 conclusions were contradicted by the executed run and are now corrected.**

- Was: "the distance at a confirmed lick is a broad distribution, not a value" and "no threshold is
  correct." Actual: median 11.0 px, 90th percentile 29.9, saturating by 50 px. A 30 px threshold
  captures 90% of confirmed licks. The distribution is concentrated, not broad.
- Was: "a fraction of lickometer licks have no tracked tongue at all ... a floor on how well any
  pose-based QC can do." Actual: **1 of 5,846** (0.0%), against 69.6% of random windows. Tracking
  is present essentially whenever the lickometer fires. This was written as a weakness; it is a
  positive result.

§5's takeaway said "nothing in this data picks one for you." It largely does: precision stays
above 0.95 out to 35 px while recall climbs 0.45 -> 0.94, and the curves cross near 35 px.

Also corrected: the cohort is **53 sessions / 17 subjects**, not the 44 / 15 in the pooled parquet
(`session_analysis_mlk` holds more). §9 limitation 2 rewritten — the pixel threshold does have a
measurable contact radius, and the live limitation is the glancing-contact bias at tight `d`;
limitation 3 replaced with the cross-session scale spread (Euclidean jaw-to-endpoint 53-95 px,
1.8x).

Unchanged: §2, left as trimmed by hand.

### `val_03_missed_licks`: §7 rewritten around `precision(d)`

**Previous version tagged `val_03-fixed-30px-baseline`** — the executed 53-session run at a fixed
30 px threshold. Return there if the curve version does not work out.

A single threshold sits on the shoulder of the contact-radius distribution. From the executed run,
the marginal fraction of newly-admitted excursions that the lickometer confirms is 0.97 (10-15 px),
0.94 (20-25), 0.86 (25-30), 0.65 (30-35), 0.11 (35-40). At 30 px the two populations — real licks
and non-contact approaches — are mixed, which is why the reported rate tracked `tracked_frac`
(spearman +0.329).

§7 now reports `precision(d)` = P(lickometer fires | tongue reached within `d`) over
`DIAG_THRESHOLDS = [10, 15, 20, 25, 30, 40]`, and ranks sessions by the miss rate at `D0 = 10 px`.
`D0` is 2.7x the `tongue_tip_center` error of 3.67 px (`pixel_error.ipynb`), the tightest distance
still above the noise floor.

- Purpose stated: **triage**, not prevalence. The output is a ranked review queue.
- `session_precision_curve` replaces `process_session_dir`; loads each session once and sweeps `d`
  inside, so the extra thresholds cost no extra I/O.
- Two panels instead of three. The old panels 2 and 3 plotted the same quantity — `pose_only_frac`
  is exactly `1 - precision`.
- The confound check is now printed rather than plotted: `spearman(miss_rate@D0, tracked_frac)`,
  against the +0.329 the 30 px version gave.
- `filter_timestamps_refractory`'s per-call print is muted in the loop via `redirect_stdout`; it
  was emitting ~106 lines and burying the per-session output.
- §8 metric A and the §10 export follow the new quantities.

On the example session the headline drops from 198 candidates (3.63%, 30 px) to 24 (0.91%, 10 px),
with a *narrower* confidence interval — [0.61%, 1.35%] vs [3.16%, 4.15%] — because the rate falls
faster than n does.

**Known limitation, unresolved.** If the lickometer fails on glancing contacts, those licks sit at
larger `d` and `D0` excludes them, biasing the miss rate low. The shape of each curve carries the
evidence; only hand-scored video settles it.

### `REORG.md` retired; `CLAUDE.md` now carries the real map of `code/`

The reorg is finished, so its planning document is gone. Two files nominally held the same
job — a map of `code/` — and the one Claude actually reads at session start was the weaker
one: `CLAUDE.md`'s "Key notebooks and scripts" described `fip_utils`/`fip_00` in detail and
said nothing about the 19 `eph_*`/`kin_*` notebooks.

**`CLAUDE.md`** gained an accurate map: all 30 notebooks as one-line questions by series
(`kin_*`, `eph_*`, `fip_*`, `val_*`, model-quality, pipeline), a table of the ten repo modules
and what each owns, and a "things worth knowing before editing" list (the two cached
intermediates, the load-bearing `eph_08`/`eph_09` spike-count windows, the absolute-vs-relative
column-name trap in the pooled parquet, and the QC definition of noise). Also documents the
flat-module import constraint, which was only recorded in `REORG.md`.

**New section on the library.** `aind-dynamic-foraging-behavior-video-analysis` was never
described in `CLAUDE.md` at all, despite being where most domain code lives and what the open
TODO items refer to. It now records the repo and local clone path, that
`environment/Dockerfile` installs it as an editable checkout pinned to `@main` (so a merge
upstream reaches the capsule on its next image build, with no version pin to shield against
it), `requires-python = ">=3.9"`, a table of its eight modules, and that its `README.md` is
still the unedited AIND template — read the source, not the README. The known
library-vs-repo boundary problem is stated there with a pointer to the `TODO.md` item.

**Two misclassifications fixed rather than copied forward.** `TODO.md` had flagged both, and
they would have been inherited verbatim:

- `model_quality.ipynb` is a single-session Lightning-Pose pipeline walkthrough, **not**
  foraging behavioral-model quality.
- `test_session_wrapper.ipynb` is a batch runner wrapping `run_batch_analysis` — operations,
  not methods evaluation. Moved to pipeline / data generation.

**New `code/archive/README.md`** carries the archive rationale that would otherwise have been
lost: what each archived notebook was superseded by, and — the part that matters — the warning
not to strip outputs from `spatial_axis_comparison_rt_encoding.ipynb`, whose stored outputs are
the only surviving record of the poster figure's provenance and are what made the replication
possible.

`REORG.md` itself is deleted; read it at `git show 52585bf:REORG.md`.

### `eph_09` MERFISH block ran — the reorg has nothing left waiting on a run

The capsule image rebuild picked up `scanpy==1.10.3` and the MERFISH structural axis ran on
Code Ocean. Two consequences recorded in `TODO.md` and `REORG.md`:

- All four items in the `eph_09` "NOT verified — needs a Code Ocean run" list are closed.
  The list is kept as a record of what was checked, not as open work.
- The `scipy==1.13.0` tripwire re-asserted in the scanpy Dockerfile layer **did not fire** —
  the build succeeded, so scanpy did not move shared packages. The tripwire stays: isolated
  `RUN` layers separate pip's *resolution*, not the environment, so a later build could still
  hit it.

`REORG.md` now states plainly that the reorg is complete and the file is a current-state map
of `code/` rather than a plan. Also cleared the last two "written, unrun" markers (`kin_07`,
`eph_09`) from the port-plan table and `kin_07`'s KEEP entry, and the stale trailing paragraph
of the execution plan, which still described three archive gates as open.

No analysis code touched.

### Docs: reconcile `TODO.md` / `REORG.md` with the completed runs

Status review after `eph_09` and `kin_07` ran on Code Ocean. Five places still described work
that is finished:

- `TODO.md` carried an **orphaned second gate table** — an earlier patch replaced the table
  header but left the old body rows, so the file listed three archive gates as still open
  directly below the table saying all five were met. Removed.
- `TODO.md` recommended-order items 4 and 6 still read "not yet run on Code Ocean" / "Code
  Ocean-only sections unrun" for `eph_09` and `kin_07`. Both now record the 2026-09-16 runs.
- `TODO.md`'s `eph_09` "NOT verified" item 2 ("Everything in `eph_09` past §1. No cell
  touching real data has run.") is struck through — the run needed none of the debugging it
  anticipated. Only the MERFISH block remains gated, on the `scanpy` image rebuild.
- `REORG.md`'s "Planned additions" still said `kin_07` and `eph_09` were unexecuted and
  holding three gates open.
- `REORG.md`'s `val_02_lickometer` entry still described it as not running and mid-repair;
  `val_03_missed_licks.ipynb` was missing from KEEP entirely.

Also corrected the notebook count: `code/` went 34 → 29 at Phase 2 close, and is at 30 now
that `val_03_missed_licks` has landed.

No analysis code touched.


### `val_03_missed_licks`: state the confidence conditioning in §4, plot densities

Both §4 distributions were already measured from the same confidence-filtered arrays —
`spout_distance` exists only on frames surviving `mask_keypoint_data` at `CONF`, and both the
lick-time and random-time measurements read those same arrays. The filtering was symmetric. Its
effect is not.

On a stand-in session, 3.4% of lick windows contain no tracked frame against 56.4% of random
windows, because the tongue is only visible when protruded. So the comparison distribution is not
"wherever the tongue happens to be" — it is "wherever a *tracked* tongue happens to be", i.e.
moments when the tongue was already out of the mouth. That is the conservative null, and the right
one, but it was described wrongly and its consequences were invisible:

- Both histograms now plot **densities**. They were raw counts at samples differing by more than
  2x after empty windows are dropped, which made the shapes not comparable.
- Both carry their **n** in the legend; only the lick distribution did.
- The **no-tracked-frame rate is printed for both**, next to the session-wide rate of frames
  passing the confidence floor. Previously only the lick rate was reported.
- §4 and its takeaways now state the conditioning explicitly, and note that a null drawn over all
  frames regardless of tracking would look better separated and mean less.

No change to any computed distance or to the threshold sweep.

### `val_03_missed_licks`: use the library's video alignment instead of a hand-rolled offset

§2 computed the session-to-video offset itself, as the per-row difference between `kps_raw_*`'s
video-relative `time` and `tongue_kins`'s `time_in_session`. The library already owns this:

- `kinematics.video_clip_utils.get_video_time(session_time, tongue_kins)` reads the offset off the
  first row of `tongue_kins`, which carries both time bases. Called at 0 it returns the offset.
- `video_alignment.session_time_to_video_time(times, offset)` applies it in §6.
- `kinematics.video_clip_utils.find_labeled_video(session_id, data_root)` locates the session's
  labeled video under `<root>/<session>*/pred_outputs/video_preds/labeled_videos/*_labeled.mp4`,
  replacing a hardcoded `video_preds_labeltest` path.

`load_session` now reads `time` and `time_in_session` from `tongue_kins` and returns the library's
offset. Checked: the library offset and the previous per-row median agree exactly on a planted
12.34 s offset, `session_time_to_video_time` matches `get_video_time` elementwise, and the
session→video→session round trip is exact. `find_labeled_video` was tested both ways — it returns
the file when the layout exists and raises `FileNotFoundError` otherwise, which the cell catches
and reports as a skip.

`video_alignment.compute_video_session_offset` is the other entry point, computing the same
quantity from the raw acquisition CSV plus the first go-cue time. It is noted in §2 but not used,
since the intermediate tables already carry both bases and the raw CSV is not in
`intermediate_data/`.

Clip cutting stays on `tongue_lickometer_utils.extract_clips_ffmpeg_encode`, which re-encodes and
so lands on the requested frame. `video_clip_utils.extract_clips_ffmpeg_after_reencode` is faster
and supports `filename_stems`, but stream-copies with `-c copy`, which snaps the start to the
nearest preceding keyframe — too coarse for a 1 s clip centered on one event at 500 fps. The
tradeoff is recorded in a comment at the call site.

### `val_03_missed_licks`: new notebook — pose tracking as QC on lickometer misses

Answers [dynamic-foraging-processing#96](https://github.com/AllenNeuralDynamics/dynamic-foraging-processing/issues/96),
"QC request for missed licks". Inverts `val_02`: there the lickometer was the reference and pose
was scored against it; here pose is the instrument auditing the lickometer, and the quantity of
interest is the **pose-only** event — tongue at the spout, no lickometer contact.

Written to read linearly for someone who does not work with this data. 35 cells, 19 code.

- **§3 what the pose data looks like** — an 8 s window of tongue x/y with spout positions, over a
  distance-to-nearer-spout trace with the threshold and lickometer ticks marked.
- **§4 distance at lickometer licks** — minimum tracked tongue-spout distance within ±100 ms of
  every lickometer lick, against a random-time null, plus a CDF annotated with what each candidate
  threshold captures. This is the figure that shows why no single pixel threshold is correct.
  Also reports the fraction of lickometer licks with no tracked tongue at all, which floors the
  method's sensitivity.
- **§5 threshold sweep** — precision and recall vs threshold, the three event counts vs threshold,
  and the two plotted against each other as an explicit operating-point curve.
- **§6 what pose-only events are** — trajectory panels for pose-only events beside matched licks
  for contrast, then labeled-video clips. Clip times convert session time back to video time via
  `VIDEO_OFFSET`, recovered from the per-row difference between `kps_raw_*`'s video-relative
  `time` and `tongue_kins`'s `time_in_session`. (`val_02` §9 cut clips at session times against a
  video-time video, so its clips are offset by that amount.)
- **§7 across all sessions** — the §5 classification over every session in `session_analysis_mlk`,
  reporting per-session rate ranked and colored by tracking quality, rate vs tracked fraction as
  the confound control, and the spread of precision.
- **§8 possible QC metrics**, characterized not prescribed: session-level rate, per-trial flag,
  a high-confidence subset cut at a low percentile of the confirmed-lick distance distribution
  (the answer to the false-positive rate in §5), and a tracking-coverage gate. §9 lists six
  limitations, including that the issue's own session has no pose output.
- **§10** writes every figure's numbers to `FIG_DIR/val_03_summary.json` for building a
  shareable write-up.

Verified: all 19 code cells parse under Python 3.9, no use-before-definition, and every cell was
executed end to end against a three-session stand-in fixture on disk — no errors, no warnings.
Locally the notebook opens clean, printing one skip line per data-dependent cell. **It has not
been run against real data**, so no number in it has been seen yet.

### `val_02_lickometer`: clear the `idxmin` FutureWarning in §8

`tongue_masked[spout_cols].idxmin(axis=1, skipna=True)` ran over every frame, including the
untracked ones where both distance columns are NaN. pandas warns that all-NA rows will raise
`ValueError` in a future version.

It was also mislabelling those frames. `idxmin` returns NaN for an all-NA row, and NaN fails the
`== 'distance_to_left_spout'` test, so the lambda fell through to `'Right'` — every frame below
the confidence threshold was labelled right-spout. `idxmin` now runs only over the rows that
have a distance, and untracked frames are left unlabelled.

Nothing downstream reads `nearest_spout`; only `nearest_spout_distance` is plotted, and that is
unchanged. Checked against the old path: identical on every tracked frame, NaN instead of
`'Right'` on untracked ones.

The one warning left when the notebook runs is a `DeprecationWarning` about Jupyter migrating to
platformdirs, raised from `seaborn`'s import chain and from `tongue_kinematics_utils`, not from
notebook code. Its fix is `JUPYTER_PLATFORM_DIRS=1` in the image, which is `/environment`'s call,
and it would fire from the kernel before any cell runs regardless. Not silenced here — a blanket
warnings filter would hide unrelated deprecations too.

### `val_02_lickometer`: name the counts for what they measure; formatting cleanup

`calculate_metrics(a, b, w)` returns `(matched, unmatched-in-b, unmatched-in-a)`. The notebook
passes pose licks as `a` and lickometer licks as `b`, so the legacy names were the wrong way
round: `fp` counted unmatched *lickometer* events and `fn` unmatched *pose* events, which made
`precision` a property of the lickometer list and `recall` a property of the pose list. Renamed
throughout to say which list each number describes. **No value changes** — verified against the
legacy formulas over 200 randomized inputs through the real library function, max absolute
difference 0.

| was | now | is |
|---|---|---|
| `tp` | `n_matched` | pose licks matched to a lickometer lick within the window |
| `fp` | `n_lickometer_only` | lickometer licks with no pose match |
| `fn` | `n_pose_only` | pose licks with no lickometer match |
| `recall` | `pose_matched_frac` | of the pose licks, fraction matched |
| `false_negative_rate` | `pose_only_frac` | of the pose licks, fraction unmatched |
| `precision` | `lickometer_matched_frac` | of the lickometer licks, fraction matched |
| `false_discovery_rate` | `lickometer_only_frac` | of the lickometer licks, fraction unmatched |
| `f1_score` | `f1_score` | symmetric in the two lists; name was already correct |
| `FP_times` | `lickometer_only_times` | lickometer licks with no pose match |
| `FN_times` | `pose_only_times` | pose licks with no lickometer match |

`n_matched + n_pose_only` is the pose lick count and `n_matched + n_lickometer_only` the
lickometer lick count; both identities are asserted in the verification above. The §3 header
states the convention once; the §8 header records that the `Status` strings in the classified
tables (`False Positive` / `False Negative`) are the library's own and keep their original
spelling. §9's output directory is now `labeled_clips/lickometer_only/`, was `false_positive/`.
Figure titles and axis labels follow the new names.

- **Removed the two missing-image references** in §3 — `threshold_example_image.jpg` and
  `examples_of_problems.jpg` under `/root/capsule/scratch/figures/`, neither present. The two
  headings they illustrated are kept as text.
- **Formatting** — consistent `# ` comment spacing; dead code dropped (`tp_rate`/`fp_rate`/
  `fn_rate`, computed and never used; an unused `import subprocess`; superseded commented-out
  plotting calls); the three near-identical interaction-plot blocks in §5 and the two ILI
  histogram blocks in §6 collapsed into loops over their differing parameter; `.copy()` on the
  FacetGrid slice in §5 to silence a `SettingWithCopyWarning`; f-strings with no placeholders
  turned into plain strings; a NumPy-style docstring on `plot_tongue_trajectory`; interpretation
  cells given consistent `### Interpretation` headers, claims unchanged.

Verified: all 17 code cells parse under Python 3.9, no use-before-definition in execution order,
and every rewritten analysis and figure cell was executed against stand-in data outside the
notebook. Setup and the guarded load cell were run locally.

### `val_02_lickometer`: verification pass on the port
Checked the ported notebook against the legacy source at `5e0bc8e:code/tongue_lickometer.ipynb`.
All 17 code cells parse under Python 3.9; imports, the `ENV` block and the guarded load cell
were executed locally (the load cell takes the `[skip]` branch). No use-before-definition across
cells in execution order. Library paths and signatures confirmed against the installed package:
`detect_licks` resolves to the 4-argument `tongue_lickometer_utils` copy, not the 5-argument
`tongue_kinematics_utils` one.

- Dropped the last mention of `load_keypoints_from_csv`, in a comment in §2 explaining why the
  original's `.insert()` was replaced by an assignment. The remaining comment states the fact:
  `kps_raw_*` already has a video-relative `time` column, overwritten with go-cue-relative
  session time.
- §8 header now records the argument-order behavior: `calculate_metrics_witheventkeys` names
  its parameters `ground_truth, detected_events` but is passed pose events first, so `fn`
  counts unmatched pose events and `fp` unmatched lickometer events. `FP_times` therefore comes
  from the lickometer frame and §9 writes to `false_positive/`. Left as written.

No change to the detection algorithm, the parameter sweep, any figure, or any metric name.

### `val_02_lickometer`: fix `cannot insert time, already exists`
`kps_raw_*.parquet` already carries a `time` column (`integrate_keypoints_with_video_time`
writes it as `Behav_Time - Behav_Time[0]`, alongside `time_raw`), so the original's
`tongue_masked.insert(0, 'time', ...)` collides. That line was correct against
`load_keypoints_from_csv`, which returns only `x`/`y`/`confidence`, and was carried over
verbatim when the data source changed. Now assigns instead, overwriting the video-relative
`time` with the go-cue-relative session time that `all_licks` is on.

### `val_02_lickometer`: original analysis, organized setup and loading
Same analysis as the previous entry's revert — the original algorithm, parameter sweep, figures
and interpretation, unchanged — put back into a sectioned notebook structure.

- **Setup (§1)** — imports on full dotted library paths; `ENV` / `IS_CO` / `SCRATCH` / `DATA` /
  `FIG_DIR` / `SAVE_FIG` block; `SESSION_DIR`, `EXAMPLE_SESSION` and `CONF = 0.8` named at the
  top instead of scattered as literals.
- **Load (§2)** — `find_session_dir` plus the per-session `intermediate_data/` parquets, behind
  an `IS_CO` guard so the notebook opens clean locally. `all_licks` and the spout means are
  computed once here rather than repeated in two analysis cells (they previously depended on
  `nwb` and `keypoint_timebase`, which no longer exist).
- **§3-§9** — the original cells, verbatim. Worked example at 30 px / 0.05 s; the three-way
  sweep over spatial threshold x refractory x coincidence window; F1 heatmaps and interaction
  plots; the refractory FacetGrid; ILI CDFs and histograms; metric curves at 30 px / 0.1 s; the
  event-key classification; distance to nearest spout; `plot_tongue_trajectory`; clip extraction.
  Metric names (`recall`, `precision`, `false_negative_rate`, `false_discovery_rate`,
  `f1_score`) as originally written.
- **Comment-only cells promoted to markdown**, wording unchanged — the parameter-search
  conclusions, the ILI reasoning, the 30 px / 100 ms motivation, and the final-stats note. These
  were code cells containing nothing but comments.
- Hardcoded `/root/capsule` paths in the clip cell now go through `DATA` / `SCRATCH`.

Nothing from the earlier redesign returns: no Implementation B, no tests, no alternative
metrics, no added figures. 32 cells, 17 of them code.

## 2026-09-16

### `val_02_lickometer`: reverted to the original notebook, library translation only
Reverted the whole redesign. The notebook is now the original `tongue_lickometer.ipynb`
algorithm and figures, with two changes and nothing else:

- **Imports** — full dotted library paths (`kinematics.tongue_lickometer_utils`,
  `kinematics.tongue_kinematics_utils`) replacing the bare module names. The runtime
  `pip install` cells are dropped; both packages are Dockerfile-pinned.
- **Data loading** — old cells 3-4 (NWB glob, `parseSessionID`, the always-`False` date filter,
  the raw Lightning-Pose CSV, `trim_kinematics_timebase_to_match`) replaced by one cell reading
  the per-session `intermediate_data/` parquets. `time_in_session` comes from `tongue_kins`;
  the keypoint is now `tongue_tip_center` (the current model's name for `tongue_tip`);
  masking stays at the original `confidence_threshold=0.8`; the `spout_l`/`spout_r` swap is
  preserved.

Every analysis cell after that is the original, verbatim: the 30 px / 0.05 s worked example,
the three-way parameter sweep, the F1 heatmaps and interaction plots, the refractory FacetGrid,
the ILI CDFs and histograms, the metric curves at 30 px / 0.1 s, the event-key classification,
the distance-to-spout column, `plot_tongue_trajectory`, and the video clip extraction.

**Removed entirely** (all of it added 2026-09-16 and reverted the same day): Implementation B
and its unit tests, the scoring-convention self-test, the directional rate wrappers, the
precision-recall surfaces and QC operating points, the confidence-premise measurement, the
event-time offset analysis, the refractory-vs-hysteresis test, the disagreement triage, the
`plotstyle` restyling, the `IS_CO` skip-guards and the section commentary. 37 cells -> 18.

The original's metric names (`recall`, `precision`, `false_negative_rate`,
`false_discovery_rate`, `f1_score`) are restored as written.

## 2026-09-16

### `val_02_lickometer` §3: simplify the loading cell
- **`time_in_session` is now read from `tongue_kins.parquet`** rather than recomputed.
  `kinematics_filter` reindexes onto its input rows, copies extra columns back by position, and
  asserts `result['time'] == df['time']`, so `tongue_kins` and `kps_raw_tongue_tip_center` are
  row-aligned. Correcting the earlier entry, which claimed the filter restricted its output to
  the tracked-frame span — that restriction applies only to the internal interpolation grid.
- **Drops the whole time-base apparatus**: the `nwb_df_trials` read, the `goCue_start_time_raw`
  column-candidate lookup, `t0`, the `raw_timestamps` cross-check and the range-overlap
  assertion. The `goCue_start_time` KeyError is moot — the trials table is no longer read.
- **Drops `REQUIRED_INTERMEDIATES` and the two explicit existence checks.** A missing file now
  raises from `pd.read_parquet` naming the path, which is the same information.
- §3 loading cell: **98 lines → 32**. The §3 markdown loses two paragraphs.
- Positions still come from `kps_raw_*`, not `tongue_kins` — unchanged, and for the unchanged
  reason: `kinematics_filter` interpolates across NaN gaps, contaminating x/y at protrusion
  edges, which is where a contact threshold is crossed. Only the *time* column is taken from
  `tongue_kins`, and that one the filter guarantees.
- Local run and mock smoke test both pass; the mock now writes a row-aligned `tongue_kins`.

## 2026-09-16

### `val_02_lickometer`: fix the `goCue_start_time` KeyError; use `find_session_dir`; drop the §2 self-test
- **Bug fix (first Code Ocean run).** `nwb_df_trials.parquet` has no `goCue_start_time` column.
  `create_df_trials(adjust_time=True)` **drops** every absolute time column and replaces each
  with `<col>_in_session` / `<col>_in_trial`, keeping the unshifted first-go-cue value as
  `goCue_start_time_raw`. The plan's A1 block (`TODO.md`) specified `goCue_start_time`, which
  never existed in that table. Now reads `goCue_start_time_raw`, accepts the pre-`adjust_time`
  name as a fallback, and raises listing the actual columns if neither is present. The value is
  printed so the time base is visible in the run log.
  *Confirms `session_analysis_mlk/<session>/intermediate_data/` exists — the item's first open
  assumption.*
- **`find_session_dir`** (`ephys.tongue_ephys`) replaces the hand-built `SESSION_DIR / session`
  path, matching `ephys_utils.py:348`. It handles the exact-match-then-prefix-glob lookup.
- ~~Considered and rejected: sourcing `time_in_session` from `tongue_kins.parquet`.~~
  **Wrong — corrected in the next entry.** The `[t_min, t_max]` restriction in
  `kinematics_filter` applies to its internal interpolation grid, not to its output; the
  function reindexes back onto the input rows, copies extra columns by position, and asserts
  `result['time'] == df['time']`. `tongue_kins` is row-aligned with `kps_raw_*`.
- **Considered and rejected: `load_intermediate_data`** for the trials/licks reads. It loads five
  tables including `tongue_kins`, `tongue_movs` and `nwb_df_events`, none of which this notebook
  uses — a large read to obtain two small ones.
- **Removed the §2 self-test cell.** The direction is resolved in code: the two wrappers convert
  `tp`/`fp`/`fn` once and nothing downstream reads them. The test asserted what those six lines
  already state, and the library change that would break it (the pending de-duplication) alters
  arity and so fails loudly anyway. The convention statement stays in the §2 markdown, now with
  the original notebook's stored counts as a scale reference for a fresh run
  (recall 0.926 / precision 0.968).
- **Still not covered by any library function**: loading `kps_raw_*.parquet`. `kin_06` has an
  inline `load_kps_raw` and this notebook now has a second ad-hoc reader — a repo-side duplicate
  worth folding into the library alongside the other pending changes.

## 2026-09-16

### `tongue_lickometer.ipynb` → `val_02_lickometer.ipynb`; markdown rewritten
- **Renamed** (`git mv`), with `FIG_DIR` and the references in `REORG.md` and `TODO.md` updated.
  `pixel_error` / `test_session_quality_analysis` deliberately not moved — that is a separate
  repo-wide decision.
- **All 18 markdown cells rewritten to state facts only.** Removed the framing prose: the
  "where this is going" section, the argument that F1 is the wrong objective, the "labelling
  trap" / "the deliverable" / "the QC product" headings, and the read-this-before instructions.
  The header is now one sentence of goal, the two lick definitions, the quantities reported, and
  scope. F1 and the two directional rates are presented as three reported quantities rather than
  as competing objectives — the sweep computes all three at every parameter combination, so both
  questions are answered from the same output.
- Same treatment applied to the print statements and docstrings: verdict blocks in §§3.2, 7, 8
  and 9 now state the outcome and its consequence without commentary.
- No code paths changed. Re-verified after the rewrite: local run (§1, §2, §4.1, §5 unit tests)
  and the synthetic-`intermediate_data/` smoke test of every Code Ocean cell both pass.

## 2026-09-16

### `tongue_lickometer.ipynb`: Implementation B, the six comparison figures, the refractory test
Second half of the `TODO.md` plan. B lives in the notebook, not the library, until it is
validated on real data; `detect_licks` stays as the baseline.

- **`detect_contacts_hysteretic`** — gap-aware hysteretic contact detection, fully vectorized
  (running maxima of the last open/close frame index; no Python loop over frames, ~0.12 s on
  1.8 M frames). Four changes to A, one per structural property: the full time base is kept so
  an untracked gap longer than `max_gap_s` terminates a contact; events are timestamped at
  closest approach rather than first entry; enter and exit use different thresholds with a dead
  band between; and confidence is gated asymmetrically (`conf_assert` to open, `conf_deny` to
  close). Fixed parameters are justified from elsewhere rather than tuned — `max_gap_s = 0.020`
  is the library's own `segment_movements_trimnans(max_dropped_frames=10)` tolerance at 500 Hz,
  `conf_assert = 0.90` is the pipeline's, `conf_deny = 0.60` is a sanity-check value whose
  sensitivity is reported as a line rather than given an axis.
- **Five unit tests for B, run locally on synthetic traces** — closest-approach timing invariant
  across entry thresholds; one event on a noisy protrusion where A emits two; immunity to A's
  re-arming collapse; gap termination on the same trace §4.1 uses for A; and agreement with an
  explicit frame loop over a 20 s noisy trace.
- **Dead-band bounds are measured, not picked.** Lower: `pixel_error.ipynb` gives
  `tongue_tip_center` = 3.67 ± 2.02 px against human labels, so the band must clear ~4-6 px.
  Upper: §3.1 computes this session's jaw→spout distance. The grid spans 5-25 px.
- **Six figures**: (1) detection counts and ILI distributions, A vs B at confidence 0.80/0.90;
  (2) the precision–recall surface with two named QC operating points; (3) event-time offset vs
  threshold; (4) the refractory-vs-hysteresis test; (5) the two disagreement populations with
  distance-to-spout triage; (6) labeled video clips.
- **Two design problems found and fixed while building, both departures from the plan as
  written:**
  - *The overlap window was free during operating-point selection.* It is a **scoring**
    parameter, not a detector parameter — letting it float lets the search buy agreement by
    widening the window until everything matches, flattering whichever detector has worse
    timing. The surface and both operating points are now evaluated at a fixed 100 ms window;
    §8 asks separately whether 100 ms is justified. The full overlap sweep is still computed,
    because §8 needs it.
  - *The flat-in-refractory test was confounded.* The refractory filter also deletes **genuine**
    fast licks, so F1 moves along the refractory axis whenever the refractory period approaches
    the real ILI, whether or not artifacts remain — the test could report "not flat" for a
    reason unrelated to hysteresis. Now the flatness check is restricted to refractory periods
    below the shortest genuine ILI (taken from the lickometer's own distribution, not assumed),
    and the direct measurement — how many events the filter still finds to delete — is reported
    and plotted alongside. Both criteria must pass, and the failure branch names the three
    candidate second sources of double-triggering.
- **§10's triage buckets are now mutually exclusive**, cut at the detector's own two thresholds,
  so each count points at a different explanation: pose blind (dropout), inside the entry
  threshold (pose should have fired), inside the dead band (near-miss), beyond it (the only row
  that is a QC finding on its own).
- Every verdict in §§3.2, 7, 8 and 9 prints both branches and states what a contradicting result
  would mean. Everything the plan asserted about B — the 14-22 ms latency offset, the
  threshold-dependent drift, the flat-in-refractory prediction — came from synthetic traces and
  is labelled as a prediction in the notebook.

**What was executed.** Locally: §1, §2, §4.1's synthetic demonstrations and §5's five unit
tests — all pass. Every Code Ocean-only cell was additionally smoke-tested end to end against
synthetic stand-in `intermediate_data/` parquets, which caught a missing `io` import, an invalid
per-bar `alpha` list, and the two design problems above. **No cell has been run against the real
session**; `data/for_local/` has only the pooled parquet. Two assumptions remain unverified and
are checked at runtime rather than assumed: that
`session_analysis_mlk/<session>/intermediate_data/` exists (the loader raises with the
missing-file list instead of falling back), and that tracking confidence is lowest during
retraction (§3.2 measures it and prints a verdict with a stated consequence for B's change 4).

## 2026-09-16

### `tongue_lickometer.ipynb`: repair (A1/A2/A3) and Implementation A
Implements the first half of the `TODO.md` plan logged earlier today. The notebook now runs;
before this it could not, because every domain import was a bare name for a module that had
moved into the library.

- **Imports (A2)** — full dotted library paths, choosing deliberately between the two modules
  that define the same six names: `detect_licks` / `calculate_metrics` /
  `calculate_metrics_witheventkeys` / `filter_timestamps_refractory` from
  `kinematics.tongue_lickometer_utils`, `mask_keypoint_data` /
  `plot_keypoint_confidence_analysis` from `kinematics.tongue_kinematics_utils`. These are the
  paths that survive the pending library-side de-duplication, so the notebook does not block on
  that PR.
- **Loading (A1)** — re-pointed at the per-session `intermediate_data/` parquets. Deleted the
  old cells 3-4 entirely: the NWB glob with its always-`False` date comparison, `parseSessionID`,
  the hand-rolled `trim_kinematics_timebase_to_match`, the raw Lightning-Pose CSV load and the
  manual time-zeroing. Also drops the dependency on the `matt_test_DLC_LP_results_20240920`
  asset. Positions come from `kps_raw_*.parquet` rather than `tongue_kins.parquet`, because the
  latter is post-`kinematics_filter` and its values are interpolation-contaminated at protrusion
  edges — exactly where a contact threshold is crossed.
- **Verified the time base rather than assuming it.** `create_df_events(adjust_time=True)` sets
  the first go cue to t=0 and keeps the unshifted values in `raw_timestamps`, so
  `time_raw - goCue_start_time[0]` puts tongue frames and `nwb_df_licks['timestamps']` on one
  base with no offset arithmetic anywhere. The loader asserts that relation and the range
  overlap, so an upstream change fails loudly instead of yielding plausible, wrong rates.
- **No fallback when `intermediate_data/` is missing** — the cell raises with the list of files
  it wanted. `kin_05`/`kin_06` assume this path but neither has been run against it, so its
  existence is still unconfirmed.
- **Scoring layer (A3), with directional names.** `score_pose_vs_lickometer` /
  `classify_pose_vs_lickometer` / `sweep_overlap_window` report
  `pose_recall_vs_lickometer` and `pose_precision_vs_lickometer`, never a bare
  `precision`/`recall`. **New §2 self-test** — hand-built events pinning down which output slot
  of `calculate_metrics` is which; it runs locally and confirms that with pose events passed
  first (as every call site does), `fn` holds pose-only events and `fp` holds lickometer-only
  events. This is the mislabelling the old cells 19-20 had backwards. Re-reading the notebook's
  own stored counts under the checked convention: `pose_recall_vs_lickometer` = 0.926,
  `pose_precision_vs_lickometer` = 0.968 (180 pose-only, 435 lickometer-only events).
- **Implementation A ported unchanged** as the baseline, with its three structural properties
  demonstrated on synthetic traces that run locally. One correction to the plan: A's re-arming
  failure is **not** unconditional. It collapses to one event per session precisely when the
  confidence mask is at least as tight as the spatial threshold (measured boundary: at a 30 px
  threshold, masking at 35 px gives the correct 4 events, masking at 30 px gives 1).
- **Checked the tracking-confidence premise** that B's asymmetric gate rests on (§3.2): written
  as a peri-apex confidence measurement with an explicit printed verdict and a stated
  consequence if it fails. Code Ocean only, so still unmeasured.
- Conventions brought in line with `kin_00`/`kin_06`/`eph_09`: runtime `pip install` cells
  removed (both packages are Dockerfile-pinned), `ENV`/`IS_CO`/`FIG_DIR`/`SAVE_FIG` block added,
  `plotstyle` applied, question-first markdown header, `IS_CO` skip-guards throughout.
- Filename left as `tongue_lickometer.ipynb`; the `val_02_spout_contact_detection` rename is a
  repo-wide prefix decision to be taken on its own, and nothing here depends on it.

## 2026-09-16

### Planning: repair and refocus `tongue_lickometer.ipynb` (plan only, no notebook changes)
- Audited `code/tongue_lickometer.ipynb` (21 cells). It does not run: every domain import is a
  bare name for a module promoted into `aind-dynamic-foraging-behavior-video-analysis`.
- **The library's two kinematics modules define the same six functions, and four have drifted.**
  `detect_licks` differs in *signature* (extra `timestamps` arg) and is **225× slower** in
  `tongue_kinematics_utils` (pure-Python row loop: 51.8 s vs 0.2 s per 1.8 M-frame session);
  `calculate_metrics` / `calculate_metrics_witheventkeys` differ in return arity, the extra
  `tn` being computed as counts subtracted from a timestamp. Call sites fail loudly against
  `tongue_kinematics_utils`, not silently. Resolution recorded in the library/repo boundary item.
- Reframed the notebook around the question it actually asks — can pose tracking detect a lick
  *defined as tongue–spout contact* — which no other notebook or library path asks; the
  pipeline's `coverage_pct` route is contact-agnostic and recall-only. Consequence: **F1 is the
  wrong objective** for the stated goal of using pose as QC on the lickometer, since the two QC
  uses want opposite precision/recall asymmetries. Plan replaces the F1 argmax with a
  precision–recall surface and two named operating points.
- Scope settled: **one session**, two detector implementations compared — the current
  threshold+refractory detector as baseline, and a gap-aware hysteretic detector timestamped at
  closest approach. Multi-session deferred as not-yet-well-posed (the threshold is in pixels,
  which are not transferable across sessions).
- Benchmarked both sweep grids (A ~38 s, B ~6 s per session) and prototyped the hysteretic
  detector against three known failure modes of the current one. All B results so far are
  synthetic and flagged as such.
- Added the plan as a `TODO.md` item; updated the `tongue_lickometer.ipynb` entry in `REORG.md`;
  appended the duplication resolution to the boundary item. Three adjacent findings
  (`model_quality.ipynb` misdescribed, `test_session_wrapper.ipynb` misfiled,
  `detect_licks_multiple` orphaned in `tongue_kinematics.ipynb`) recorded as explicitly
  **out of scope** for that item.

### Reorg Phase 2 complete: all five HOLD notebooks ported and archived
- `eph_09` and `kin_07` verified on Code Ocean by the user; `eph_08` had already replicated
  the reference poster figure (r=0.184, p=0.0681, n=99). That cleared the last gates.
- `git mv`'d all five into `code/archive/`: `tongue_latency`,
  `tongue_kinematics_ephys_intertrialmovs`, `spatial_axis_comparison_rt_encoding_update`,
  `tongue_kinematics`, `tongue_kinematics_cueresponse`. `code/` goes from 34 to 29 notebooks.
- Checked first that nothing in `code/` executably depends on any of the five — every
  apparent reference was the library module `tongue_kinematics_utils` (a substring match) or
  a provenance comment.
- Confirmed `spatial_axis_comparison_rt_encoding_update` is fully superseded: `eph_09` covers
  its §1-§4, §6-§9 and §9b; §5's machinery is `spatial_axes.py`; the abs-`T_rt` panel is
  `eph_08` §5; §12's commented registry sketch is answered by eph_09's proper registry use.
- `TODO.md` gate table and `REORG.md` HOLD / execution-plan sections rewritten to record
  completion. This closes the HOLD-port work begun 2026-09-11.

### Docs: record the reference-figure replication
- `TODO.md`: replaced the stale "eph_08 output equivalence … not verified" item (and the
  earlier claim that the target was unreproducible) with a "Replication of the reference
  figure" subsection — where the figure came from, the two root causes (windows, ML fold),
  what was deliberately not matched (the mesh), and why the archived notebook's stored
  outputs and `execution_count` values were the whole case. Superseded the window values in
  the 2026-09-15 `all_counts_df` fix note. (Two further items raised here — eph_09 §8's
  signed `T_rt` and the absence of an anatomical filter outside eph_08/eph_09 — were
  reviewed and dropped 2026-09-16.)
- `REORG.md`: corrected `spatial_axis_comparison_rt_encoding.ipynb`'s ARCHIVE entry — it is
  **not** an "old dup of `_update`" but the notebook that generated the poster figure, and
  the only surviving record of its provenance. Updated both HOLD rows.

### eph_08 / eph_09: fold ML to the positive side (second replication fix)
- After the window fix, `eph_08` produced r=0.19, p=0.0637, n=99 against the reference's
  r=0.184, p=0.0681, n=99 — n exact, r off by ~0.006.
- Cause: the ML fold sign. Both notebooks folded unit coordinates to `-ML`
  (`ccfs[:, ml] = -np.abs(...)`) while the structural axes are *fitted* in `+ML` space
  (`ccf_wf[:, ml] = np.abs(...)`), flipping the ML component of every projection relative to
  the axis it is projected onto. The reference folds to `+ML` ("POSITIVE, matching upstream").
  Isolated locally: `+ML` gives r=0.0130 and `-ML` gives r=0.0230 on the cached table, a
  +0.010 shift matching the observed gap in sign and magnitude.
- Flipped to `np.abs` in `ccf_points_lps_mm` (parameter renamed `fold_left` -> `fold_right`,
  all call sites updated), the mesh centroid, and `eph_09`'s §7 arrow origin.
- Kept the newer `20250418_transformed_remesh_10_ccf25.obj` mesh rather than the reference's
  `new_core_mesh.obj`. Centering is a pure translation along the axis, so this leaves r, p
  and n unchanged and only offsets the plotted x-axis (~0.19 mm vs the reference figure).
  Verified the mesh is entirely `+ML` (0.519-1.311 mm, 0 of 40208 vertices negative), so
  `eph_09`'s §7 contours are unaffected by the fold change.


### eph_08 / eph_09: use the reference run's spike-count windows and z_ccf filter
- Traced the poster figure `rt_response_projection_abs.svg` (2026-05-04 00:34:41) to
  `code/archive/spatial_axis_comparison_rt_encoding.ipynb` cell 42 (`execution_count` 37),
  committed 16 min later in `2f20188`. Confirmed by the SVG's per-panel tick colours, which
  are that notebook's `COLORS` entries for the waveform / MERFISH / retrograde axes, and by
  cell 42's stored output matching the figure's r/p/n exactly.
- **Root cause of the failed replication: the spike-count windows.** Cell 9 of that run
  (`execution_count` 7) builds `all_counts_df` with `count_window_s=(0.0, 0.5)` and
  `baseline_window_s=(-2, 0.0)`. `eph_08`/`eph_09` had `(0.0, 0.2)` / `(-1.0, 0.0)`,
  inherited from a commented-out cfg in `spatial_axis_comparison_rt_encoding_update.ipynb`
  — a file created *after* the figure, in the same commit, whose commented block had already
  been changed. The 500 ms / 2 s windows also match the published analysis.
- Changed both notebooks to `(0.0, 0.5)` / `(-2.0, 0.0)`; both continue to build
  `all_counts_df` in-notebook. Added a markdown note before each counts cell recording the
  provenance and a numeric check (response window should give 50 nominally / 43 FDR
  significant units of 103; 42 / 36 means the old windows are still in effect), plus prints
  of those counts.
- Added the reference's `z_ccf` ∈ [-5.2, -3.5] anatomical filter with its own note. It drops
  exactly one mislocalised unit (`behavior_758017_2025-02-06_11-26-14` unit 85,
  `z_ccf = -2.174`, 2.25 mm from the mesh centroid vs 0.79 mm for every other unit), taking
  the projection from n=100 to the reference's n=99. In `eph_09` it is placed after the axis
  fits and before §8, matching the reference's ordering (axis fit n=100, projections n=99).
- Verified locally against `data/for_local/`: with the old windows the pipeline reproduces
  the current `eph_08` output exactly (r=0.011, p=0.9109, n=100, proj max 2.196,
  |T_rt| max 12.096), and reproduces the reference's mesh centroid to 8 decimals, its
  unit counts, and its waveform axis — isolating the windows as the only remaining
  difference. Rebuilding counts needs spike times, so the final check must run on Code Ocean.

## 2026-09-15 (8)

### `kin_07_value_encoding.ipynb` — two fixes from the first Code Ocean run

- **Fixed a `KeyError: 'max_x_from_jaw_y'` in §5.** `build_trial_tables` selected only
  `KIN_COLS`, so the column §5 derives `max_x_from_jaw_y_rel` / `_distance` from never
  reached `trials_cue`. Added `AUX_KIN_COLS` — carried into both trial tables but
  deliberately excluded from §6's screen grid, since the raw absolute version is a
  near-duplicate of `max_y_from_jaw` and would add a redundant heatmap row.
- **Replaced the RidgeCV "beats null by 2 SD" criterion with a permutation p-value.**
  Found by the new integration test below: on fabricated noise latents, `summarize_screen`
  reported `rpe` as beating its null on an R² of +0.0000 against a null of −0.0024. With
  3 shuffles the null SD is itself noise, so the criterion was meaningless. `ridge_r2_screen`
  now defaults to 20 shuffles and returns `p_perm` = `(1 + #{null >= r2}) / (1 + n_shuffles)`;
  a latent counts as predicted only if R² > 0 **and** `p_perm` < 0.05, starred on the panel.
  §8's checklist item 3 updated to say so — this feeds the notebook's headline conclusion.
- **`run_value_screen` gained a `methods` argument.** §6.3 reads only the Spearman panel but
  was paying for the random forest and both nulls; it now requests `("spearman",)`. Skipped
  panels come back NaN-shaped so the plot/summary helpers still accept the result.
- Added a loud warning when `add_jaw_relative_columns` resolves zero jaw keypoints, instead
  of silently producing all-NaN columns, and guarded §5's Wilcoxon the way `spearman_screen`
  already was.

**New verification.** `verify_pipeline.py` fabricates a `trial_latents` table with the real
schema from the local parquet and runs every stage §3.6 onward — `build_trial_tables`,
`add_jaw_relative_columns`, `per_session_slopes`, `run_value_screen` (all four methods and
spearman-only), `add_previous_trial_rpe`, `plot_value_screen`, `summarize_screen`. It
validates *column availability at each stage*, not any scientific result, and is what would
have caught the `max_x_from_jaw_y` bug before the Code Ocean run. All stages pass; the
synthetic-latent screen test still passes unchanged.

## 2026-09-15 (7)

### `kin_07_value_encoding.ipynb` — new: do behavioural-model latents explain tongue kinematics?

Sixth and last of the HOLD ports. Merges the kinematics-vs-model-latents analysis that
existed in two partly-overlapping single-session copies — `tongue_kinematics.ipynb` cells
111-138 and `tongue_kinematics_cueresponse.ipynb` cells 17-28 — into one pooled notebook.
No `kin_*` or `eph_*` notebook previously used behavioural-model latents against
kinematics (`eph_06` uses them against ephys), so this is a topic the refactored series
did not track at all.

**Status: written, executed locally only to the extent the data allows.** §6.1 (the
kinematic covariance structure) runs on the pooled parquet and has real output; everything
from §3.4 on needs per-session `nwb_df_trials.parquet` and is Code Ocean-only, so it is
written and reasoned through but unexecuted. The notebook was run end-to-end locally to
confirm every gate takes its skip path cleanly (no errors, 23 code cells).

**Dependency check, done before writing anything.** `get_mle_model_fitting` still resolves,
and the fetch is fast enough that pooling was never in question:

- The import path both sources use —
  `aind_analysis_arch_result_access.han_pipeline.get_mle_model_fitting` — is now a
  **deprecated shim** forwarding to `...df_mle_model_fitting.get_mle_model_fitting`. The
  notebook prefers the new path and falls back to the shim, so it survives the shim's
  removal. `environment/Dockerfile` installs the package unpinned, so Code Ocean tracks
  whatever is current — which is the argument for not importing through the shim. No
  Dockerfile change needed (and none made).
- **Coverage, measured over all 44 sessions in ~48 s:** 40 sessions have a
  `QLearning_L2F1_CKfull_softmax` fit; the four sessions of subject **751004**
  (2024-12-20 … 2024-12-23) have no MLE records at all and drop out.
- Latent shapes confirmed: `q_value` and `choice_kernel` are `(2, N+1)`, `choice_prob`
  `(2, N)`, `rpe` `(N,)`.

**Pooled, not single-session.** Both sources run on one session. At ~1 s/session the fetch
is not the cost driver, and pooling is what makes **session** available as a sampling unit
— which is the whole point, given the statistics below. 40 sessions.

### Three statistical problems in the sources, not inherited

- **Pseudo-replication.** Both sources merge trial-level latents onto **movement**-level
  rows (`tongue_with_q = tongue_movements.merge(...)`), so *n* becomes the movement count
  while the predictor only varies per trial. Measured inflation in this dataset: 246,359
  movements against ~14,000 trials. Fixed two ways together — aggregate to one row per
  trial (`trials_cue`, the cue-response movement; `trials_mean`, the per-trial mean over
  all movements), **and** make session the unit of inference (statistic computed within
  session, tested across sessions). The one-cue-response-movement-per-trial assumption is
  checked rather than assumed: 14,138 trials carry one, 9 carry two (0.06%), and the first
  is kept.
- **Multiple comparisons.** §6's grid is 24 kinematics × 11 latents = 264 tests;
  uncorrected, ~13 would clear p < 0.05 with nothing present. Every p is BH-FDR corrected
  across the whole grid (`multipletests(method="fdr_bh")`), matching `encoding_methods.py`
  and `per_unit_stats_registry.py`. Noted in the notebook: a Wilcoxon on ~40 sessions has
  a p-value **floor** (~8e-13 at n=44), so strong and very strong effects report the same
  q — rank by |ρ|, not by q.
- **Correlated features.** Quantified rather than asserted, in §6.1 (which runs): median
  |ρ| among the 276 kinematic pairs is a mild 0.27, but 21 pairs exceed 0.7 and **five
  exceed 0.99**. Within-session z-scored, `max_x` / `max_x_from_jaw` / `max_x_distance` are
  literally the same variable (ρ = 1.000), as are `time_to_endpoint` / `out_duration`;
  `endpoint_y` / `max_y_from_jaw` sit at 0.996. The prose next to the 2×2 figure says
  plainly that RF importance splits credit arbitrarily across such blocks and MI
  double-counts them, and that **only the RidgeCV R² panel is a genuinely predictive
  measure** — it uses all kinematics jointly and is scored on held-out sessions
  (`GroupKFold`, the same discipline `kin_06` §8 uses).

### Other decisions worth recording

- **`TODO.md` was wrong that `attach_model_latents_to_trials` is defined identically in the
  two sources.** They differ in the sign of `q_diff` — `cueresponse` cell 13 computes
  `R_value − L_value`, `tongue_kinematics` cell 115 computes `L_value − R_value` — which
  flips the sign of every correlation involving it. Ported once, as **`R_value − L_value`**,
  so that positive `q_diff` and positive `endpoint_y` in `kin_06`'s jaw-centred frame both
  mean "toward the animal's right". The second difference: `tongue_kinematics` derives
  `q_diff_c` from `animal_response` inside the function while `cueresponse` has that block
  commented out and derives it later from the movement's `event`; the `animal_response`
  version is used, being defined for every responded trial.
- **Restored the alignment assertions both sources commented out.** A length mismatch
  between the latent arrays and the trial table would otherwise shift every latent by an
  unknown offset without raising. Because the upstream API does not document which trials
  it fits, the function checks the fit length against *both* candidates (responded trials
  vs all trials) and records which matched in a `latent_index_mode` column, so a mixture
  across sessions is visible rather than silent.
- **Added two checks of the left/right row convention (§3.5), which neither source makes.**
  Both assume row 0 of each `(2, N)` array is left; a swap would negate `q_diff` and turn
  `chosen_prob` into 1 − `chosen_prob` without erroring. Check 1 compares mean
  `chosen_prob` against the fit's own reported `prediction_accuracy` (under a swap it would
  match 1 − that instead); check 2 asserts mean `q_diff_c` > 0. Both fall out of quantities
  the fit already reports.
- **`q_sum` is new, not ported.** `TODO.md` §3 asks for it and neither source has it. It is
  the natural control for `q_diff`: `q_diff` is about which option is better, `q_sum` about
  how good the environment is overall — the vigour-like quantity a motor effect would most
  plausibly track.
- **§5's four pinned pairs are framed as descriptive, and §6.3 tests them.** The source
  picked those pairs after looking at the data and attached p-values that ignore the
  search. §6.3 re-runs the screen with the two jaw-derived columns appended and reports
  each pinned pair's q and its rank within the full FDR-corrected grid, so "do the pinned
  pairs fall out of the screen" is answered by the notebook rather than by the reader.
- **`rpe_prev` is computed on the complete trial table before the merge**, not after.
  Shifting after a merge that drops trials would make `rpe_prev` the RPE of the previous
  trial *that had a cue-response movement*, which can be several trials back. The sources
  happen to get this right; it is made explicit because the ordering is easy to invert.
  §7 screens both `trials_cue` and `trials_mean`, since "how the animal moves on the next
  trial" includes the preparatory non-lick movements `kin_05` establishes are real.
- **No jaw-reconstruction code duplicated from `kin_06`.** `kin_06` §2.1 carries an
  algebraic fallback because that notebook must run locally. `kin_07` cannot run locally at
  all, so it reads the real `kps_raw_jaw.parquet` keypoint directly (`get_jaw_y`, 6 lines)
  and the ~100-line fallback is deliberately not copied.
- Dropped from the port as redundant or superseded: `tongue_kinematics` cells 119-134
  (one-off `plot_kinematics_vs_q` calls, subsumed by the §6 screen), cells 131-134
  (hexbin/scatter of kinematics against each other, no latents involved — that is `kin_01`
  territory), and `cueresponse` cells 17, 22-24 (a commented-out plot helper and the
  keypoint loading `kin_06` already owns).

### Verification

The model-fit path cannot run here, so the §6 screen machinery was tested **independently
of the real latents**: real kinematics from
`data/for_local/all_tongue_movements_04022026.parquet` (14,039 cue-response trials, 44
sessions) against a *synthetic* latent built as a known linear function of one kinematic
column plus noise, at two noise levels, with a pure-noise negative control. The driver was
chosen to be a hard target — `excursion_angle_deg` sits at ρ = 0.92-0.97 with `endpoint_y`,
`min_y`, `max_y` and `max_y_from_jaw` — so this tests discrimination against near-twins,
not just against unrelated columns. Result: **all four methods ranked the planted driver
first at both noise levels**; the noise control gave driver q = 0.99 and held-out
R² = −0.001; RidgeCV ordered strong (0.80) > weak (0.31) > noise (−0.001). The test
execs the notebook's own cell sources, so it exercises the shipped code rather than a copy.

That run also produced the number quoted in §6's prose: a latent planted on **one**
kinematic column made **17 of 24** columns clear FDR in the Spearman panel. Sixteen were
not findings — which is the correlated-feature caveat, measured.

### Files

- `code/kin_07_value_encoding.ipynb` — new, 40 cells (8 sections).
- `TODO.md`, `REORG.md` — `kin_07` moved out of planned; both HOLD rows updated.

## 2026-09-15 (6)

### `kin_06_lick_geometry_choice.ipynb` — jaw position now read from real keypoints, preferentially

Three follow-ups to the `kin_06` port (previous entry), landed interactively after publishing it.

- **Stopped overclaiming validation of the jaw reconstruction.** The notebook's own §2.1
  prose said its fit residual "confirms the reconstruction" — it doesn't. That residual is
  an internal self-consistency check (how tightly each row's two `max_y_from_jaw ±
  max_y_distance` candidates cluster), not a comparison to the true jaw keypoint, and can't
  catch a systematic offset. Corrected in the notebook (§2.1/§3 prose, docstrings, print
  statements) and in `TODO.md`/`REORG.md`, which repeated the same overclaim.
- **§9's example-session panel now reuses `EXAMPLE_SESSION`** (the same session §3/§4 use,
  and the source `cueresponse` notebook's own session) instead of auto-picking whichever
  session has the most paired trials, so one example recurs through the notebook.
- **Investigated a user-flagged anomaly** in §9's pooled scatter (a few cue-response
  endpoints on the "wrong" side of the midline; the changed-mind marginal shifted
  negative). Traced both to real, non-bug causes: (1) the marginal shift is exactly the
  sign-flip definition applied to an already-known asymmetry — right-lick trials flip side
  25.1% of the time vs left-lick's 10.4%, so changed-mind trials skew toward the
  "flipped-to-right" case (753 of 1,185); (2) a handful of individual right-lick trials
  (10 of 3,003, 0.33%) do cross the fitted midline, concentrated in sessions with unusually
  small right-lick excursion magnitude. That surfaced a bigger, previously unflagged
  finding: **41 of 44 sessions show `|left-lick excursion| > right-lick excursion`** from
  the fitted jaw, by a median of 13 px (up to 57 px) — plausibly real spout geometry (right
  spout closer to the jaw's rest position for most animals), but only checkable against the
  true keypoint, not from the pooled parquet alone.
- **Checked whether a better local jaw estimate existed before reaching for Code Ocean.**
  Neither `startpoint_x/y` nor the movement bounding-box columns (`min_x/y`, `max_x/y`) sit
  at a fixed point — within-session SD 11-39 px, all far noisier than the algebraic
  reconstruction's own ~0.1 px internal residual. Confirms the reconstruction was already
  the best available *local* estimate; the open question was never precision, it was
  whether that recovered constant is the *same* point Code Ocean's `kps_raw['jaw']` calls
  the jaw.
- **New `load_jaw_from_keypoints()` + `get_jaw_positions()`** (§2.1). `get_jaw_positions`
  now tries every session's real `kps_raw_jaw.parquet` mean first and only falls back to
  `estimate_jaw_position()`'s algebraic reconstruction where that file is missing — locally,
  all 44 sessions, since no keypoint data exists in `data/for_local/`. Where both exist, it
  reports the gap between them directly, so the fallback's accuracy is validated
  automatically across every available session the first time this runs on Code Ocean,
  rather than spot-checked on one example the way §3's old cross-check did. `estimate_jaw_
  position()` itself is unchanged — it's now explicitly documented as the fallback, not the
  primary path. Re-executed end-to-end locally: prints "0 of 44 sessions from the real
  keypoint, 44 from the algebraic fallback," all downstream numbers unchanged.

## 2026-09-15 (5)

### New `code/kin_06_lick_geometry_choice.ipynb` — does pre-lick tongue direction predict choice?

Phase 2 of the repo reorg: the fifth of the six HOLD-notebook ports. Source is
`tongue_kinematics_cueresponse.ipynb` cells 19, 49-56, 93-95 plus `tongue_kinematics`
cell 60. `tongue_kinematics_cueresponse.ipynb` now gates on `kin_07` alone.

**35 cells, §1-§9.** Structured so the landmark figures read as the setup (§3-§4) and the
choice decode as the result (§7-§8), per the `TODO.md` outline. Executed end-to-end
locally against `data/for_local/all_tongue_movements_04022026.parquet` (246,359 movements,
44 sessions); §3-§4 print skip messages and are unexecuted.

**Correction to `TODO.md`'s data-availability note: `max_*_from_jaw` is NOT jaw-relative.**
`max_x_from_jaw` / `max_y_from_jaw` are the *absolute* pixel coordinates of the point
farthest from the jaw; the jaw-relative distances are `max_x_distance` / `max_y_distance`
(pooled means ~368 px vs ~25 px). So those columns are **not** pooled-safe as written, and
neither are `endpoint_x/y` — per-session mean `endpoint_y` spans ~80 px across the 44
sessions against a ~51 px within-session SD. Pooling them raw would have smeared the
anatomical frame the whole notebook rests on.

**The jaw origin turns out to be recoverable from the pooled parquet** — no Code Ocean
needed. New `estimate_jaw_position()` (§2.1) solves for each session's jaw keypoint from
the absolute/distance column pairs: `jaw_x = median(max_x_from_jaw - max_x_distance)`
(the tongue protrudes in +x, so the minus branch is correct — within-session SD ~4 px vs
~20 px for the plus branch), and `jaw_y` by a 1-D search over the two per-row candidates
`max_y_from_jaw ± max_y_distance`. Worst-session residual 0.25 px across all 44 sessions —
**but that residual is an internal self-consistency check, not external validation.** It
measures how tightly each row's two candidates cluster around the fitted point; it cannot
detect a systematic offset (e.g. `jaw_y` skewed toward whichever spout draws more excursions).
The real check — comparing against the actual `kps_raw['jaw']` keypoint mean — is written into
§3 but has **not executed**, since no `kps_raw_*.parquet` exists in `data/for_local/`. Treat
the reconstructed `jaw_y` as unverified until that cell runs on Code Ocean.
Everything from §5 on works in the resulting jaw-centered `*_rel` frame. This is what lets
§5-§9 pool at all while §3-§4 stay Code Ocean-only: the **spouts** still need per-session
keypoint means (`kps_raw_*.parquet`); the jaw is the only landmark recoverable without them,
and even that recovery is pending confirmation.

- **§3-§4 (Code Ocean only, unexecuted)** — `plot_standard_lick_landmarks` (source cell 19,
  the only copy in the repo) brought in here and restyled onto `plotstyle`; **not**
  promoted to `plotstyle.py`, which is style-only and has no second consumer. Plus the
  jaw↔spout scale printout (cell 94), a cross-check of the §2.1 reconstruction against the
  true keypoint mean, and cue-response endpoints coloured by lick side over the landmarks
  (cell 93). The source's mirrored spout naming (`spout_l` is the animal's *right* spout —
  the bottom camera flips left/right) is preserved and documented.
- **§5 — lick vs non-lick excursion geometry.** `tongue_kinematics` cell 60 **==**
  `cueresponse` cell 95; **ported once**, here, as `TODO.md` specifies. Pooled and
  jaw-centered, as 2-D density rather than the source's alpha scatter (246k points
  saturate where the source's ~7k did not). With lick 65.3 ± 10.9 px from the jaw vs
  40.8 ± 15.8 px without; per-session paired medians 66.2 vs 39.3 px, Wilcoxon p = 1.1e-13
  (n = 44 sessions).
- **§6 — non-lick endpoints by movement ordinal** (cells 54, 56). The source referenced two
  names (`nonlick`, `licks`) it never defined — leftovers from another notebook;
  reconstructed here as the obvious `has_lick` split.
- **§7 — does pre-lick direction predict choice?** (cells 49, 55). 7,138 trials, 44
  sessions; class balance **57.9% left / 42.1% right** (majority-class accuracy 0.579).
  Last pre-lick excursion angle: median -60.0° before a left lick vs +38.5° before a right
  lick (session-paired Wilcoxon p = 1e-11, n = 44). Binned P(right lick) rises monotonically
  from 0.08 to 0.94 across `endpoint_y_rel`, crossing the base rate within one bin of the
  jaw midline, and 0.09 → 0.88 across `excursion_angle_deg`.
- **§8 — ridge-logistic decode** (cells 44, 47, 48). **The source's evaluation could not be
  reused.** It calls `train_test_split(random_state=42)` on one session's trials; applied
  to pooled data that leaks session identity across the split. Replaced with two
  evaluations, reported side by side because they answer different questions:
  - **`GroupKFold`, session as the group (primary)** — held-out-session AUC
    **0.835 ± 0.100** (SD over 5 folds), pooled out-of-fold AUC 0.836, accuracy 0.808.
    Asks whether *one* geometry→choice mapping generalizes across animals and rigs.
  - **Per-session fits (secondary)** — median AUC **0.943** over the 39 sessions with
    n ≥ 40 and ≥ 10 per class; 97% above 0.5, Wilcoxon p = 7.3e-12. Higher than the shared
    decoder, which is what session-specific camera geometry predicts — the reason both are
    reported.
  - **Shuffled-label null**, labels permuted *within* session over 200 permutations: mean
    0.530, 95th pct 0.545, observed 0.835, p = 0.005 (the permutation resolution floor).
    The null sits above 0.5 because within-session shuffling preserves session base rates,
    which a pooled evaluation can exploit; that offset, not 0.5, is the right reference.
  - **Permutation importance** (on held-out sessions, not held-out trials): `last_angle`
    0.168, `mean_distance` 0.098, `mean_angle` 0.070, `mean_duration` 0.035 ΔAUC; both
    peak-velocity terms and `session_time` ≈ 0. Direction carries the decode, not vigor.
- **§8.4-§8.5 — the block-structure caveat, quantified rather than asserted.** Left/right
  choice in this task is block-driven, so a lick-side decoder may be reading recent choice.
  Measured: **P(stay) = 0.910** and **previous choice alone reaches AUC 0.907** — *better*
  than kinematics. But kinematics still reads AUC 0.815 / 0.772 *within* each
  previous-choice stratum, so it is not simply re-encoding choice history. Not resolved
  here; separating them needs block-aware regressors, which is `kin_07`'s dependency.
- **§9 — pre-lick → cue-response displacement** (cells 50-52). Pre-lick and cue-response
  endpoints land on the same side of the jaw midline in **84.0% ± 2.2%** of trials (SEM,
  session as the sampling unit), with the cue-response lick carrying the tongue 15-23 px
  *further* from the midline (Wilcoxon p < 1e-300 both sides). Per-session
  r(pre-lick y, cue-response y) median 0.776.

Other port decisions:

- **Every grouping is on `(session, trial)`**, not `trial` — trial numbers repeat across
  sessions in the pooled parquet. Same class of bug as the one fixed in `kin_05` §7.
- **Session is the sampling unit for every population claim** — §5's paired excursion
  medians, §7's violin panel and binned curves (binned *within* session on pooled quantile
  edges, then mean ± SEM *across* sessions), §8's grouping, §9's same-side fraction.
- Dropped from the source throughout: hardcoded hex colors, seaborn violin/`palette`
  defaults, `plt.grid(True)`, and the absolute `/root/capsule/scratch/figures` save paths.
  All 8 figures end with `save_fig(..., fig_dir=FIG_DIR, save=SAVE_FIG)`.
- Label joins use merges rather than `pd.MultiIndex.from_frame(...).map(...)`, and §7's
  binning uses integer bin labels rather than Interval categoricals — both to keep pandas
  off its object-hashing path, which emitted a `RuntimeWarning` on the NaN-bearing `trial`
  column. Notebook now executes warning-free.
- `fmt_p()` reports SciPy p-value underflow as `< 1e-300` rather than printing `0`.

Deliberately **not** done: no `git mv` of `tongue_kinematics_cueresponse.ipynb`. Unlike
`eph_07` / `eph_09`, `kin_06` does not clear its source on its own — that notebook is gated
on `kin_06` **and** `kin_07` (the value-encoding port), so it stays in `code/`.

## 2026-09-15 (4)

### New `code/spatial_axes.py` + `eph_09_structural_axes.ipynb`; `eph_08` refactored onto the module

Phase 2 of the repo reorg: the fourth of the six HOLD-notebook ports. Source is
`spatial_axis_comparison_rt_encoding_update.ipynb` cells 13-38, which is now fully
replicated and (pending a Code Ocean run) becomes archivable.

- **New `code/spatial_axes.py` (~700 lines).** Fitting and comparing 3-D spatial *axes* —
  the direction along which a feature varies fastest. `fit_spatial_axis_linear` / `_cca` /
  `_LDA` (scalar / multivariate / categorical features) with bootstrap wrappers,
  `compare_bootstrap_directions` (tangent-plane Wald test), `cone_half_angle`,
  `vectors_to_az_el`, `plot_projected_arrow_with_cone`, `get_regression_CI` and
  `plot_projection_scatter`. Ported from source cells 14, 15, 31 with function bodies
  unchanged. Kept separate from `spatial_encoding.py` deliberately: that module asks
  *where* a statistic is large (CCF maps, permutation tests) and has no axis-fitting
  machinery; this one asks *in which direction* it changes.
- **`eph_08_waveform_axis.ipynb` refactored onto it** — its two private inline copies of
  `fit_spatial_axis_cca` / `bootstrap_spatial_axis_cca` (51 lines) deleted in favor of
  `from spatial_axes import bootstrap_spatial_axis_cca`. Structured as a pure move: no
  other change, because its projection figure is the poster figure
  `rt_response_projection_abs`. **Not verified against real data** — `eph_08` is Code
  Ocean only and skips locally. What *was* verified: the module and the deleted inline
  copies produce bit-identical axes, bootstrap clouds, cone half-angles and downstream
  projections on synthetic waveform-shaped input (same seed, same RNG draw sequence).
- **New `code/eph_09_structural_axes.ipynb` (34 cells).** Asks whether the RT-encoding
  spatial gradient is the *same* gradient as LC's structural ones.
  - Fits the RT-encoding axis (`T_rt`) and its baseline control (`T_rt_bl`) —
    **the step `eph_08` skipped**; `eph_08` projects onto the waveform axis but never
    fits the RT axis, so there was previously nothing to compare against.
  - Three structural axes, each behind its own `HAS_WAVEFORM` / `HAS_MERFISH` /
    `HAS_RETRO` guard so a missing asset drops that axis from every downstream
    comparison rather than failing or being substituted.
  - Pairwise direction comparison (angle, Wald W, chi2 p, bootstrap p) + summary table,
    three-plane arrow-and-cone figure, azimuth-elevation bootstrap scatter, and
    projection scatters of `T_rt` / `T_rt_bl` onto each structural axis.
  - Carries over the source's interpretation guide, extended with the two failure modes
    the summary table hides (small angle + significant p; large angle + wide cone).
  - Source cell 40 (a commented-out registry sketch) dropped. Instead the RT statistics
    are computed through the real machinery — `AnalysisSpec` / `fit_encoding` /
    `PerUnitStatsRegistry`, as `eph_01` does — so `T_rt` is the same quantity
    `eph_01`-`eph_04` report and no third inline copy of `build_rt_encoding_stats` was
    created. Session/unit QC likewise goes through `data_loading` rather than being
    reimplemented inline as `eph_08` does.
  - **One deliberate departure from `eph_01`:** no RT trial window (`trial_query=""`,
    `min_trials=50`), matching the source notebook and `eph_08` so the fitted axes
    describe the same units the poster figure projects. `RT_QUERY` is left in place for
    a sensitivity check.
- **Retrograde LDA sign convention preserved** — an LDA axis sign is arbitrary, so it is
  pinned to the waveform axis as the source does, and the notebook says so explicitly
  (and warns when no waveform axis is available to pin against).
- **`environment/Dockerfile`: added `scanpy==1.10.3`** in its own layer, following the
  `rachel-analysis-utils` precedent. The MERFISH axis cannot run without it, and it was
  absent from the pip block. 1.10.3 is the last scanpy supporting python 3.9 (1.10.4+
  require >=3.10); `scipy==1.13.0` is re-asserted alongside it as a tripwire so a future
  scanpy bump fails the build loudly instead of silently moving numpy/scipy under the
  other pins. **This triggers a Code Ocean image rebuild.**

**Verification status.** `spatial_axes.py` is pure numpy/sklearn math and was tested
locally against synthetic data (30 checks, all passing): `fit_spatial_axis_linear`
recovers a planted gradient to 0.00 deg noise-free and 3.4 deg at heavy noise;
`_cca` and `_LDA` recover planted axes to ~1-4 deg; `compare_bootstrap_directions`
gives 0.00 deg / p=1.0 for identical axes, 0.25 deg / p=0.98 for two samples of the same
axis, and 89.7 deg / p<0.002 for orthogonal ones; `cone_half_angle` widens monotonically
with noise (0.46 -> 2.49 -> 6.43 -> 20.26 deg) and as n falls (5.3 -> 13.2 deg).
`eph_08` and `eph_09` both execute clean locally through their skip paths, but **neither
has been run on Code Ocean**, so no real-data output — including `eph_08`'s poster-figure
equivalence — has been confirmed. See `TODO.md`.

## 2026-09-15 (3)

### Extract shared `fip_*` setup into `code/fip_utils.py` (Pass 1 of 2)

- **New `code/fip_utils.py`.** `fip_00_explore.ipynb`, `fip_01_movement_value_coding.ipynb` and
  `fip_02_ne_only_events.ipynb` each carried their own copy of the same load -> curate ->
  session-setup preamble (283/248/241 non-blank lines, ~490 of them pure duplication). Every bug
  fix this cycle had to be applied 2-3 times by hand; the `regex=False` substring fix alone
  touched 7 call sites. The module follows the existing `plotstyle.py`/`ephys_utils.py`
  convention (flat module in `code/`, bare import, no `sys.path` juggling) and carries the
  loading/curation, session select, `parse_event`/`get_trace`/`build_meta`/`pick_example`,
  trial enrichment, motion-energy-on-the-FIP-clock, signal helpers, and the multi-session
  `process_session` pipeline. Notebook code drops 1657 -> 893 lines (-764).
- **Memory fix is now structural.** `load_curated_sessions` holds the pre-curation `nwb_list_raw`
  and the unused second curated list as function locals, so both are released on return. The
  `del nwb_list_raw, _nwb_list_curated_unused; gc.collect()` incantation is no longer something
  each notebook has to remember.
- **Two switches onto upstream functions** (everything else is a move, not a change):
  - `fu.enrich_trials` calls `rachel_analysis_utils.analysis_utils.enrich_df_trials`, with a
    local two-column reimplementation only as a fallback. This replaces *two* separate
    reimplementations — `fip_00`'s `enrich_streaks` (which never tried upstream) and `fip_01`'s
    `_enrich_streaks_and_rpe_bins_fallback`. The reason `fip_00` avoided upstream ("won't parse
    on Python 3.9, nested-quote f-string at `analysis_utils.py:294`") is stale: that file is 177
    lines and `python3.9 -m py_compile` clean. The fallback keeps `fip_00`'s `ses_idx` grouping,
    which `fip_01`'s bare `.shift(1)` lacked and which multi-session needs so streaks don't run
    across session boundaries.
  - Onset detection consumes the upstream `data_z` column
    (`fu.threshold_onsets(..., already_z=True)`) instead of z-scoring internally, so one
    normalization convention runs throughout. `fu.zscore` switches to `ddof=1` to match
    `enrich_dfs.zscore_fip` exactly — verified equal to
    `scipy.stats.zscore(x, ddof=1, nan_policy="omit")` to 4e-16, and the `ddof` change moves zero
    samples across threshold at n=100k, so onset times are unaffected.
- **`attach_me_to_df_fip` convention reconciled to raw ME.** `fip_00`'s multi-session copy stored
  `me_z` while `fip_01`/`fip_02` stored raw. Raw is correct: `fip_psth_inner_compute` takes a
  `data_column` argument (default `"data"`) and never z-scores internally, so raw-in-`data` plus
  `data_column="data_z"` is the composable form, and the `enrich_dfs` functions z-score `data`
  themselves. Only affects `fip_00`'s `BUILD_NWB_LIST_ME=False` cell, which prints guidance.
- **`fip_00`'s internal duplication resolved** — `build_meta`, `locate_me_assets` and
  `attach_me_to_df_fip` were each defined twice, once in the single-session cells and again in
  the multi-session helpers cell. The multi-session cell keeps only the four functions Pass 2
  will replace (`session_etr_mean`, `aggregate_series`, `collect_region_etr`, `plot_by_subject`)
  and imports the rest by name, leaving the four analysis cells and results table untouched.
- **`fip_01` sheds a dead helper cell** — `zscore`/`threshold_onsets`/`peri_event` became unused
  there when Figure 3 moved to `fip_02`.
- Each notebook's imports cell gains `%load_ext autoreload` / `%autoreload 2`: a session load is
  tens of GB and minutes, so without it every `fip_utils.py` edit would cost a kernel restart
  and a full reload.
- **Upstream curation API has been rewritten — Dockerfile pin needed.** At
  `rachel-analysis-utils` main (`b7b7487`, "CSV-based data curation inputs"),
  `apply_curation_nwb_list` no longer exists; `data_curation_helpers.py` is now
  `load_curation` + `apply_curation_df_fip`, reading CSVs via
  `aind_bwnm_fiber_data_curation_utils` — imported at module top level but declared in no
  packaging metadata (`dependencies = []` at main as well as at the pin), so pip installs
  nothing for it and the import raises `ModuleNotFoundError` — while overwriting
  `df_fip['event']` with target names and dropping `intended_measurement` entirely. `environment/Dockerfile:60` installs `@main`
  unpinned; the cached image still has the old API, so the next rebuild would break the curation
  cell and then `parse_event`/`build_meta`/`pick_example`. Recommended pin
  (`864550d55356ecb4906d05801cdae694bd000fde`) is written up in `code/fip_todo.md` — not applied
  here, since CLAUDE.md puts `/environment` off limits.
- Verification is static only (CO-only data assets): `python3.9 -m py_compile` on the module,
  `nbformat` validate + `nbconvert --to script` + `compile()` + `pyflakes` (no undefined names,
  no shadowed module functions) on all three notebooks. Pass 1 has **not** been run on Code
  Ocean yet.

## 2026-09-15 (2)

### Create `eph_07_bout_encoding.ipynb` — bout helpers consolidated into `ephys_utils.py`
- **Bout helpers ported into `ephys_utils.py`.** `annotate_movement_bouts` (previously the sole
  copy, living only in `tongue_kinematics_ephys_intertrialmovs.ipynb` cell 10) is relocated
  unchanged. The within-trial/ITI classifier — copy-pasted across four near-identical
  `get_session_bout_times` cells in the source, with drifting thresholds (`2.0/2.0/1.0` at cell
  12 vs `1.0/2.0/0.5` at cell 29) — is consolidated into two functions: `classify_bout_times`
  (the shared dt-to-nearest-go-cue logic, generalized to take arbitrary bout onset times) and
  `get_session_bout_times` (wraps it with `annotate_movement_bouts` for the movement-bout
  definition), both with the thresholds as explicit parameters rather than hardcoded. Verified
  locally against `data/for_local/all_tongue_movements_04022026.parquet` (44 sessions, 246,359
  movements → 31,081 bouts; median bout size 5, row-weighted mean 17.9; pinned thresholds
  `GAP_THRESHOLD_S=0.5, GO_RESPONSE_WINDOW_S=2.0, ITI_MIN_POST_CUE_S=2.0,
  ITI_MIN_PRE_NEXT_S=1.0` give 14,368 go-responsive and 9,297 ITI bouts pooled across sessions).
- **`eph_07_bout_encoding.ipynb`** ports the movement-bout-derived within-trial vs ITI ephys
  comparison from `tongue_kinematics_ephys_intertrialmovs.ipynb`, following `TODO.md`'s port
  plan: movement-derived bouts as primary (doesn't inherit the lickometer's blind spot for
  non-lick movements, per `kin_05`'s finding), lick-bout-derived (`licks["bout_start"]`) as a
  lighter-weight robustness check in §9. Consolidated the four near-identical population-PETH
  cells (source 14/15/19/22) into one pass, keeping the session-wide z-scoring normalization
  (source cell 15). Per-unit go-responsive-vs-ITI Δ firing rate comparison (source cells 29–31)
  registered through `encoding_methods.fit_encoding` + `per_unit_stats_registry` (binary `is_iti`
  predictor, OLS) so it composes with `eph_01`–`eph_04` rather than standing alone as a one-off
  Wilcoxon test. Dropped source cell 37 (video-clip extraction — a one-off, and the clip helpers
  are library-owned).
- Reused `eph_00`'s `make_rp_and_events`/`compute_psth`/`smooth_vector`/`plot_psth` raster/PSTH
  path throughout (single-unit example in §5 and the population loop in §6), rather than the
  source's hand-rolled `peth_for_unit` — the population loop still builds a `RasterPlotter` per
  unit × condition, at a coarser 20 ms bin size to keep the loop over hundreds of units cheap
  (matching the source's own reason for that bin-size choice, not a reason to avoid the shared
  path).
- **Only §4 (the bout-helper verification) executed this session** — everything from §5 onward
  needs per-session spike times and intermediate data that exist only on Code Ocean. Written and
  reasoned through, syntax-verified (the notebook was actually executed end-to-end locally; every
  Code-Ocean-only cell took its `ENV == "local"` skip branch without error, which at minimum
  confirms every such cell parses), but not run — none of it should be treated as a finding until
  it runs there.
- **Flag, not acted on:** `CHANGELOG.md` already had a `## 2026-09-15` section (now below,
  unnumbered) placed *after* the `## 2026-09-14 (9)`…`(2)` block instead of above it — out of the
  file's stated newest-first order. Left as found; worth a cleanup pass if noticed independently.
- Not archived this session: `eph_07` is `tongue_kinematics_ephys_intertrialmovs.ipynb`'s sole
  remaining port gate (see `TODO.md`/`REORG.md`), but archiving is a separate step after review.

## 2026-09-14 (9)

### `fip_02_ne_only_events`: add a z-threshold sensitivity sweep
- New closing section: reruns onset detection and all four quantitative NE-DA
  analyses (shuffle control, lag distribution, peak-amplitude correlation, ME
  alignment) across `Z_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0]`, one four-panel summary
  figure per threshold, to check how much of the main analysis depends on the
  choice of detection threshold. Ends with a small per-threshold summary table
  (onset counts, coincidence rate + p-value, median lag, Pearson r).
- Refactored as `analyze_at_zthresh(z_thresh)` returning a dict, rather than five
  copies of the module-level analysis cells — reuses the existing `threshold_onsets`,
  `peak_in_window`, `peri_event`, and `coincidence_rate` helpers unchanged; only
  `z_thresh` varies across the sweep, `REFRACTORY`/`MIN_RUN`/`WINDOW` stay fixed at
  the notebook's main values. Shuffle count reduced to 500 (from 1000) per sweep
  point since the whole pipeline reruns five times.
- Not executed — same Code-Ocean-only data assets as the rest of the `fip_*` series.
  Validated via `nbformat.validate` read-back and `ast.parse`/`py_compile` on the
  converted script.

## 2026-09-14 (8)

### `fip_02_ne_only_events`: add raw-trace and cross-correlation views ahead of onset analysis
- Two new sections inserted right after the z-scored NE/DA/ME traces are built, **before**
  onset detection — a qualitative continuous-signal look precedes the discrete-event analysis
  that follows it:
  - **NE/DA raw traces over a window** (`TRACE_WINDOW_DUR`, default 10s, plus a
    `plot_ne_da_window(t_start, duration)` helper to scan other parts of the session) — overlay
    the two z-scored traces to get a visual sense of how correlated they look before
    quantifying anything.
  - **NE↔DA cross-correlation**: resample both onto a common 20 Hz grid over their time overlap
    (same recipe `fip_00_explore.ipynb` uses for motion energy × FIP) and run `norm_xcorr`
    (added to the Helpers cell, same function `fip_00` defines). Sign convention deliberately
    set to match the onset-lag distribution further down the notebook (positive = NE leads DA).
- No changes to the onset-detection/classification/quantitative sections below — this is a
  qualitative + continuous-signal complement to the existing analyses, placed to build the
  notebook's narrative forward (raw look → continuous relationship → discrete events →
  event classification → event-level quantification → behavioral comparison).
- Not executed — needs the same Code Ocean-only data assets the rest of the `fip_*` series does.
  Validated via `nbformat` read-back and `ast.parse`/`py_compile` on the converted script.

## 2026-09-14 (7)

### Remove Figure 3 from `fip_01`; new `fip_02_ne_only_events` — NE/DA transient dissociation
- **`fip_01_movement_value_coding.ipynb`**: removed Figure 3 ("NE"-only onsets aligned to
  movement) — it treated every PL-GCaMP ("NE") onset alike, which is now a distinct notebook's
  question. Also removed the now-unused `NE_EVENT` selection and trimmed the title/example-
  signals markdown accordingly. `fip_01` is left with its original two figures (movement per
  RPE bin; z-scored AUC per consecutive R-/R+), both still looped over all example FIP channels.
- **New `code/fip_02_ne_only_events.ipynb`**: which NE (PL-GCaMP) transients happen *without* a
  concurrent DA (NAc dLight) transient ("NE-only") versus with one ("NE+DA"), and does movement
  (motion energy) look different around the two kinds of NE event. Classification: independent
  `threshold_onsets()` on NE and DA (same parameters used throughout this notebook family), an
  NE onset counts as NE+DA if a DA onset falls within ±0.5 s of it, otherwise NE-only — reuses
  the existing onset-detection machinery symmetrically rather than a separate DA-magnitude rule.
  Four analyses quantify the relationship beyond that binary split: DA z-score peak distribution
  around every NE onset (with `z_thresh` marked, to show whether the split is a real dichotomy
  or a continuum); a circular-shift shuffle control for the coincidence rate (empirical p-value
  against each signal's own baseline event rate); lag distribution for coincident pairs (does DA
  lead/lag NE); and NE-vs-DA peak-amplitude correlation (Pearson + Spearman) for coincident
  pairs. A QC figure (example NE-only vs. NE+DA traces) sits before those. Main figure: ME
  aligned to each onset group, mean ± SEM overlaid.
  - Only needs continuous-time `data_z` (via `aind_dynamic_foraging_data_utils.enrich_dfs.
    zscore_fip`, same as `fip_01`) — deliberately skips `fip_01`'s per-trial
    `enrich_fip_in_df_trials`/`remove_tonic_df_fip` pipeline, since this analysis is onset-based,
    not per-trial. Setup cells (data load, curation + memory fix, session select, `pick_example`,
    motion-energy alignment) copied from `fip_01`, same as `fip_01` copied from `fip_00` — now a
    3rd notebook duplicating this setup; `fip_todo.md` updated to flag the shared-module
    extraction as overdue.
- Not executed — needs the same Code Ocean-only data assets the rest of the `fip_*` series does.
  Validated via `nbformat` read-back and `ast.parse`/`py_compile` on both notebooks; confirmed
  no leftover `NE_EVENT`/Figure-3 references remain in `fip_01`.

## 2026-09-14 (6)

### Create `fip_01_movement_value_coding.ipynb` — movement × RPE/value coding
- New notebook, built on the example session `fip_00_explore.ipynb` loads (`SESSION_IDX=0`,
  subject 808054). Extends the analysis used for the FIP-only phasic/tonic-DA poster figure
  (baseline AUC by consecutive-reward streak, outcome traces by RPE bin) to motion energy (ME)
  and to the "NE" channel (this dataset's curation has no nLight sensor; PL-GCaMP stands in for
  NE per project convention). Three figures: (1) session-averaged movement per RPE bin, via
  `aind_dynamic_foraging_basic_analysis.plot.plot_fip.plot_fip_psth_compare_alignments` on an
  ME pseudo-channel, aligned to choice and split by `RPE-binned3` — same function/window/color
  convention as the reference pipeline's `PAC_2026.ipynb` recipe; (2) z-scored AUC per
  consecutive R-/R+, via `aind_dynamic_foraging_data_utils.enrich_dfs.enrich_fip_in_df_trials`
  + `remove_tonic_df_fip` (Rachel's actual baseline/tonic-normalization functions — not a
  reimplementation), paired with `num_reward_past` shifted by one trial (the exact pairing
  convention traced from `power_analysis.ipynb`/`foraging_summary_plots.py::plot_baseline_corr`,
  easy to get backwards); (3) "NE"-only onsets aligned to movement — no analog in Rachel's
  pipeline (she never analyzed motion energy), reuses `fip_00_explore.ipynb`'s own
  `threshold_onsets`/`peri_event` machinery restricted to the PL-GCaMP channel.
- Traced the recipe through `rachel-analysis-utils`, `aind-dynamic-foraging-basic-analysis`,
  `aind-dynamic-foraging-data-utils`, and (for plotting conventions only, not a dependency)
  `AllenNeuralDynamics/DA_phasic_tonic`. Full trace, function-by-function, and a dedicated
  review of deviations from Rachel's actual pipeline (what was fixed vs. inherent since ME was
  never part of her analysis) are in the implementation plan for this notebook.
- Incidental finding while tracing this: `rachel_analysis_utils.analysis_utils`'s
  Python-3.9-incompatible nested-quote f-string (cited in `TODO.md`'s Python-3.11-upgrade
  section) appears fixed upstream — a fresh clone of `main` compiles cleanly under
  `python3.9 -m py_compile`. Not yet verified on Code Ocean; logged in the new `fip_todo.md`
  (see below) rather than acted on.
- **New file `fip_todo.md`**: a short to-do list scoped to the `fip_*` series, separate from
  `TODO.md`/`REORG.md`'s `kin_*`/`eph_*` port-plan tracking. Seeded with the `fip_utils.py`
  extraction (this notebook currently duplicates `fip_00_explore.ipynb`'s setup cells rather
  than sharing a module), the `analysis_utils` Python-3.9 re-verification above, and the
  "anticipatory movement during CS+Delay" figure that was requested alongside these three but
  deferred (no literal CS+/delay epoch exists in this dynamic-foraging task).
- Not executed — needs the same Code Ocean-only data assets `fip_00_explore.ipynb` does.
  Validated via `nbformat` read-back and `ast.parse`/`py_compile` on the converted script only.

## 2026-09-14 (5)

### Create `kin_05_nonlick_movements.ipynb` (Phase 2 port, item 2/6)
- New notebook: do the lickometer and video-derived movement streams describe the same
  events, and what are the movements that aren't licks? Framed as a definitional
  question, not a QC filter — lickometer agreement, confidence, and duration are not
  valid noise criteria (that's `kin_00`'s job). §3 three-way correspondence tally
  (licks w/o movements, movements w/ >1 lick, movements w/o licks), §4 licks without
  movements, §5 movements with multiple licks, §6 kinematic profile of non-lick
  movements (extends `kin_01` §5 to `duration`/`total_distance`, not a duplicate of its
  `out_peak_velocity`/`out_duration`), §7 per-trial lick vs. non-lick structure, §8
  preparatory timing (prevalence of non-lick movements before the cue-response lick +
  donut of trials by pre-lick movement count). §9 (illustrative single-trial/raster
  figures) already landed in `kin_02` §11 last session — not re-ported.
- **Correction to the port plan: §3 needed a partial Code-Ocean-only split, not just
  §4.** `TODO.md` labeled §3 "pooled and locally testable" with only §4 flagged
  Code-Ocean-only. Checked directly against the source (`tongue_kinematics` cell 35):
  part (a), "licks without movement," computes
  `nwb.df_licks['nearest_movement_id'].isna().sum()` — a per-session column, confirmed
  absent from all 49 columns of `all_tongue_movements_04022026.parquet`. Parts (b) and
  (c) use `tongue_movements['lick_count']`/`['has_lick']`, both pooled. Split §3
  accordingly: (b)/(c) compute and print locally (verified: 8,585/151,404 = 5.67%
  movements with >1 lick; 94,955/246,359 = 38.54% movements without licks); (a) moved
  under the existing §4 Code-Ocean guard (written, unexecuted, with the expected
  `nwb_df_licks.parquet` schema documented).
- **Second plan gap, same category:** `TODO.md`'s §5 spec ("cells 43, 44 ... pooled")
  is only true of cell 43 (the `lick_count > 1` tally, already covered by §3b — not
  re-run). Cell 44, the multi-lick example-trace figure, needs per-frame `tongue_segmented`
  and per-session `nwb.df_licks` — Code-Ocean-only, same as §4. Given the tally is
  already the pooled result and the per-session example is a qualitative aside on an
  already-rare event (~0.03–32% by session, median ~2.9%), didn't port it as a second
  CO-only guarded block; ported a pooled `lick_count` distribution + per-session rate
  instead, which is the actual population-level finding cell 44 can't provide from a
  single session.
- **Bug fix in the port, not present in the source:** the source's per-trial structure
  (cells 64-68) grouped by `trial` alone, correct only because that notebook runs on one
  session. `trial` numbers repeat across sessions in the pooled parquet, so §7 groups by
  `(session, trial)` instead — silent cross-session pooling would have inflated the
  per-trial movement counts.
- §8 uses the pooled `movement_before_cue_response` column directly rather than
  reconstructing it from `nearest_movement_id` (what the source did) — the reconstruction
  needs the per-session lick table and is unnecessary since the flag is already pooled.
- Verified end-to-end locally against `data/for_local/all_tongue_movements_04022026.parquet`
  (246,359 movements, 44 sessions) via `jupyter nbconvert --execute`; 0 errors, 6 figures
  rendered. Only §4 (licks without movements) is unexecuted, Code-Ocean-only.
- `tongue_kinematics.ipynb` archiving now gates on `kin_07` alone (`kin_05` gate met).

## 2026-09-14 (4)

### `fip_00_explore`: drop the `video_alignment`-branch workaround, use `read_video_csv`
- **Cleanup, no behavior change on the current dataset.** `aind-dynamic-foraging-behavior-
  video-analysis`'s `video_alignment` module (previously only on a `video_alignment` branch)
  is now merged to `main`, and the Dockerfile already installs `@main` (`environment/
  Dockerfile:48`). Section 8b's import cell no longer needs to git-fetch/checkout the
  `video_alignment` branch inside the editable install at `/src` — replaced with a plain
  `import ... as va`, falling back to a pip install from `@main` only for a from-scratch env.
- **Use the package's own CSV reader.** `video_alignment` now ships `read_video_csv` (auto-
  detects the Old/flat headerless camera CSV vs. the New/AIND headered layout, which names the
  behavior-time column `ReferenceTime` instead of `Behav_Time`) plus the `DEFAULT_COLUMNS` /
  `TIME_COLUMN_ALIASES` constants. `motion_energy_to_session` now calls `va.read_video_csv`
  and resolves the behavior-time column from `va.TIME_COLUMN_ALIASES` instead of hand-rolling
  `pd.read_csv(..., header=None, names=CAM_COLUMNS)` with a hardcoded `cam["Behav_Time"]`
  lookup — so it no longer silently mis-reads a New/AIND-layout CSV as headerless data.
- Verified `video_alignment.py`'s public function signatures (`compute_video_session_offset`,
  `get_first_frame_behavior_time`, `behavior_time_to_video_time`, `video_time_to_session_time`)
  are unchanged from what the notebook already called. Not executed locally (needs Code Ocean
  data assets); validated via `nbformat` read-back and `ast.parse` on the converted script.

## 2026-09-14 (3)

### `kin_02_latency` §11: fix raster rendering artifact and illegible legends
- **Rendering bug.** Both §11 raster figures (movement type, movement ordinal) showed
  spurious horizontal gaps — bands of trials with no visible ticks. Root cause: at the
  notebook's default `figure.dpi` (110, from `plotstyle.apply_style()`), a 6×6 in raster
  with ~570 trial rows renders at ~649×649 px, under 1.2 px/trial; matplotlib's
  anti-aliasing drops some trial rows unevenly at that density. Confirmed directly —
  extracted the actual embedded PNG bytes from the executed notebook, reproduced the
  identical banding at 649×649 px, and confirmed it disappears entirely by ~1180×1180 px
  (dpi 200). Fixed by passing `dpi=220` explicitly to `plt.subplots()` in both raster
  cells (scoped to just those two cells, not a `plotstyle.py`-wide change — the other
  figures in this notebook aren't dense enough to need it).
- **Illegible legend.** Both raster legends sat directly over dense scatter data with no
  background (`plotstyle`'s global `legend.frameon=False` convention), making them
  unreadable. Added `frameon=True, facecolor="white", edgecolor="none", framealpha=0.9`
  to both legend calls — a scoped exception to the frameless convention for these two
  data-dense figures specifically.
- Re-executed end-to-end; both figures confirmed banding-free with readable legends.

## 2026-09-14 (2)

### `kin_02_latency`: fix §3–§7 ordinal filter; add §11 single-session illustration
- **Correctness fix.** §3's filter never restricted to `movement_number_in_trial ==
  cue_response_movement_number` — since `cue_response_movement_number` is a trial-level
  constant, grouping by k pooled in every movement from a k-labeled trial, not just the k-th
  one. Only ~11% of the rows plotted in §4 (and consumed by §5–§7) were actually the
  cue-response movement (verified: 6,958 / 57,925 at k=1). Split §3 into `movements_valid`
  (all movements in a trial with a valid k — what §8's Δt estimate needs) and `df` (only the
  cue-response movement itself, one row per trial — what §4–§7 use); re-pointed §8's `lat_df`
  at `movements_valid`. Re-executed end-to-end: §4–§7's numbers changed substantively (e.g.
  k=1 log-normality n: 57,925 → 6,958; medians now cleanly spaced ~0.13–0.66 s across k=1–4),
  §8–§10 were unaffected since they already carried their own independent restriction.
- **New: §11, single-session illustration.** Ported the example-trial and raster/histogram
  figures from `tongue_latency.ipynb` cells 5, 8, 9, 14 (one example session,
  `behavior_716325_2024-05-31_10-31-14` — the session the source notebook itself used):
  a trial raster colored by movement type (cue-response lick / other lick / non-lick), a
  raster colored by movement ordinal (viridis, cue-response movement outlined), and a
  colored-histogram panel (lick vs. 1st/2nd-move latency, and lick latency by k reusing §8's
  `k_colors` so the same k means the same color throughout the notebook). The single-trial
  tongue y-position trace (source cell 5) needs per-frame `tongue_kins.parquet`, not in the
  pooled parquet — written with the `ENV`/`kin_00`-style guard but Code Ocean only, unexecuted.
  Checked `kin_00`/`kin_01`/`kin_03` first for duplicates: none (`kin_00`'s rasters are
  unrelated QC/spatial-radius figures; `kin_01`/`kin_03`'s "raster" hits were the
  `rasterized=True` matplotlib flag, not raster plots).
  - This content was previously slated for `kin_05_nonlick_movements` (not yet built);
    reassigned here since it fit naturally as `kin_02`'s closing illustration and the user
    asked for it directly. Added a `coerce_bool` helper (`tongue_latency` cells 6/8) since
    `cue_response` is object-dtype with `True`/`False`/`None` in the pooled parquet.
  - **Consequence:** `tongue_latency.ipynb`'s entire audited unreplicated-content list (RT +
    IMI decomposition, single-trial example, both rasters) is now ported. Its archive gate no
    longer includes `kin_05` — it has no remaining port gate. Not archived this session (out
    of scope); `TODO.md`/`REORG.md` updated to reflect this so a future session (or the user)
    can `git mv` it directly.
- Updated `TODO.md` (item body, overview table, archive-gate table) and `REORG.md` (HOLD table,
  `kin_02` KEEP note, planned-additions table) to match.

## 2026-09-14

### `kin_02_latency`: RT + IMI decomposition (Phase 2 port, item 1 of 6)
- Appended §8–§10 to `kin_02_latency.ipynb`, porting the "nice story" left behind in
  `tongue_latency.ipynb` (cells 15, 17, 18, 26): reaction time modeled as a first-movement
  latency plus a sequence of inter-movement intervals, RT ≈ RT₁ + (k−1)·Δt.
  - §8 estimates Δt as the median within-trial inter-movement interval, then de-shifts
    `lick_latency` by `(k−1)·Δt` per cue-response movement ordinal k and shows the
    distributions partially collapse onto k=1 (hist + KDE overlay).
  - §9 runs KS tests of each de-shifted k against de-shifted k=1.
  - §10 checks whether the spread of `lick_latency` grows with k (bootstrap 95% CI on SD),
    then the cross-session population version (grand mean ± SEM by k, session as the
    sampling unit).
  - Carried over the source notebook's caveat as closing markdown: the residual mismatch at
    k=1/k=2 is attributed to further covert preparatory movements, stated as an open question
    that `kin_05_nonlick_movements` (not yet built) is meant to test.
- Made the `lick_latency` (conditioned on `cue_response_movement_number`) vs
  `movement_latency_from_go` (what §4–§7 already plot) distinction explicit in prose, per
  `TODO.md`'s note — these are different quantities on different event streams.
- Flagged and fixed a latent issue in the source: `tongue_latency` cell 17's "raw vs aligned SD"
  comparison compares a quantity to itself (shifting a group by its own constant offset cannot
  change that group's SD) — kept both columns in the results table for continuity but dropped
  the redundant duplicate line from the plot and added an explanatory note.
- Fully local-testable; executed end-to-end against
  `data/for_local/all_tongue_movements_04022026.parquet` (44 sessions) with no new dependencies.
  Δt ≈ 0.172 s; KS tests reject full collapse at k=2/3/4 (p ≪ 0.001), consistent with the
  carried-over caveat.
- Updated `TODO.md` (item marked done, archive-gate table) and `REORG.md` (moved the entry out
  of "Planned additions" into the `kin_02_latency` KEEP note, updated the HOLD table and
  execution-plan checklist). `tongue_latency.ipynb` archiving still waits on
  `kin_05_nonlick_movements`.

## 2026-09-15

### Reorg planning refinements
- Dropped the planned `bout_utils.py`. `annotate_movement_bouts` and the within-trial/ITI
  classifier now fold into the existing `ephys_utils.py` — which already holds behavior-derived
  features that serve ephys alignment (`build_trial_features` takes movs/licks/trials and touches
  no spikes), so bout segmentation is the same category. `eph_07` already imports from
  `ephys_utils`, and a module with one consumer isn't worth the file. `spatial_axes.py` remains
  the only new module planned. Rationale recorded in `TODO.md`'s `eph_07` item.
- Added a `TODO.md` item: **define the boundary between the library and this repo's `code/`
  modules.** Both sides carry "kinematics utils" and "ephys utils" with no stated rule, which
  causes: duplicate definitions inside the library (`tongue_lickometer_utils` and
  `tongue_kinematics_utils` share six identically-named functions); no home rule for plotting
  (the library ships ~10 plot functions that predate and ignore `plotstyle.py`, so every port
  restyles by hand); undocumented layering between `tongue_ephys.py` and `ephys_utils.py`; and
  an undeclared cross-boundary contract on `tongue_quality_stats.json`. Item proposes criteria
  to agree and record in `CLAUDE.md`, and should be batched with the outbound-metrics
  consolidation since both need a library PR + pin bump.

## 2026-09-11

### Repo reorganization — Phase 2 planning (HOLD notebook port plan)
- Audited all five HOLD notebooks cell-by-cell (`tongue_latency`, `tongue_kinematics`,
  `tongue_kinematics_cueresponse`, `tongue_kinematics_ephys_intertrialmovs`,
  `spatial_axis_comparison_rt_encoding_update`) against the current `kin_*`/`eph_*` series to
  determine what is genuinely unreplicated.
- Found the leftover content clusters into five scientific questions that cut *across* notebooks
  rather than mapping one-notebook-to-one-port; rewrote the plan around question-oriented target
  notebooks: `kin_02` §8–§10 (RT + IMI decomposition), `kin_05_nonlick_movements`,
  `kin_06_lick_geometry_choice`, `kin_07_value_encoding`, `eph_07_bout_encoding`,
  `eph_09_structural_axes`, plus modules `bout_utils.py` and `spatial_axes.py`.
- Two findings that changed the previous plan: (1) kinematics × behavioral-model latents
  (Q values, RPE) is a topic no `kin_*`/`eph_*` notebook covers, present in *two* HOLD notebooks,
  so it gates two archivals; (2) `tongue_kinematics_cueresponse` is a choice-prediction analysis,
  not only "spatial geometry". Also confirmed `eph_08` never fits the RT spatial axis itself, and
  that `spatial_encoding.py` has no axis-fitting machinery.
- Recorded known cross-notebook duplication (port-once cases), content already covered by
  `tongue_lickometer.ipynb` (do not port), orphan code needing a home
  (`annotate_movement_bouts` — verified absent from the library on all three branches;
  `plot_standard_lick_landmarks`), pooled-parquet column availability (drives local vs Code
  Ocean-only sections), a recommended work order, and per-notebook archiving gates.
- Rewrote `TODO.md` (overview section + six work items) and updated `REORG.md` (planned
  additions, audited HOLD table, execution plan). No code changed.

### Repo reorganization — Phase 1 (archive superseded notebooks)
- Reviewed every notebook/script in `code/` against the refactored `eph_*`/`kin_*`/`fip_*` series
  and shared modules; recorded the full classification in `REORG.md`.
- Created `code/archive/` (+ `code/archive/reference/`) and `git mv`'d 15 files there (renames,
  history preserved): superseded pipelines/notebooks (`rt_ols_registry_pipeline`,
  `registry_usage_example.py`, `umap`, `tongue_kinematics_ephys`, `tongue_kinematics_ephys_figures`,
  `spatial_axis_comparison_rt_encoding` old dup), library-promoted/dev scratch
  (`old_functions_tongue_kinematics`, `cue_response_lick_processing_example`,
  `tongue_segmentation_test`, `batch_clips`, `create_labeled_clip`, `example`,
  `extract_tongue_kinematics`, `event_timeline`), and a collaborator reference notebook
  (`F_ephys_behavior_action&outcome` → `archive/reference/`).
- Tagged `wild-prereorg` as a restore point before moving anything.
- Kept in place: active analysis + shared modules, data-generation/pipeline infra, model-quality
  evaluation notebooks, and 5 notebooks with not-yet-replicated content held for porting.
- `TODO.md`: logged the outstanding ports (RT+IMI latency story, ITI bout ephys, MERFISH/retrograde
  spatial axes, lick↔movement correspondence, cue-response geometry) and the outbound-metrics
  library consolidation.

## 2026-09-07

### Repo housekeeping
- Added `TODO.md` for deferred work. First entry: upgrading the Code Ocean environment off
  Python 3.9, which is the shared root cause of the `--ignore-requires-python` layer in the
  Dockerfile, the un-importable `rachel_analysis_utils.analysis_utils` (and the local
  `enrich_streaks` reimplementation that works around it), and the repo-wide 3.9 syntax rule.
- `.gitignore`: ignore `*.code-workspace`. VS Code multi-root workspace files point at
  machine-specific paths (e.g. `../.venv/src/...`) and are not portable to Code Ocean.

### Upstream: `video_alignment` merged to main
- `aind-dynamic-foraging-behavior-video-analysis` merged the `video_alignment` branch to `main`
  (PR #2). The module is now on `main`, which `environment/Dockerfile` already tracks.
- **Follow-up, not yet done:** rebuild the Code Ocean environment, then delete the runtime
  `git fetch`/`git checkout` bootstrap in `fip_00_explore.ipynb` section 8b — it exists only
  because `main` previously lacked `video_alignment.py`, and becomes a no-op after the rebuild.
  Worth adding `fastparquet` to the Dockerfile pip block at the same time, to retire the other
  runtime `pip install` bootstrap in the imports section.

## 2026-06-30

### fip_00_explore.ipynb — multi-session comparison
- Added a "Multi-session comparison" section that reruns the single-session pipeline over all
  curated sessions and pools results with session as the sampling unit (mean ± SEM), grouped
  by region × subject. The single-session cells are unchanged and still serve as a detailed view.
- New helpers: `process_session` (per-session enrich + ME load + onsets), `build_meta`,
  `locate_me_assets`, `attach_me_to_df_fip` (injects motion energy as a `df_fip` pseudo-channel
  `event="ME"` so the upstream `plot_fip` PSTH machinery can treat it like a FIP channel), plus
  cross-session aggregation helpers (`session_etr_mean`, `aggregate_series`, `iter_region_signals`,
  `collect_region_etr`, `plot_by_subject`, `window_mean`, `streak_go_cues`).
- Four cross-session analyses: (1) ETR of FIP from ME onsets, (2) ETR of ME from FIP transients,
  (3) within-trial (0–2 s) vs ITI (2–4 s) relative to go cue, (4) peri-go-cue responses binned by
  consecutive rewards/failures via `rachel_analysis_utils.analysis_utils.enrich_df_trials`
  (`num_reward_past`). Scalar summaries collected into an in-memory `df_results`.
- `peri_event` gained an optional `censor_times` passthrough (additive) so streak/ITI go-cue
  subsets are censored against the full go-cue set.
- Sessions missing ME/video assets are skipped and logged, not fatal.
- 3.9 fix: dropped the `rachel_analysis_utils.analysis_utils` import (that module has a
  nested-quote f-string at line 294 that doesn't parse on Python 3.9, the CO env). Reimplemented
  the only piece we use — `num_reward_past` — as a local 3.9-safe `enrich_streaks` helper
  (verified to match the package's definition exactly).

## 2026-06-29

### fip_00_explore.ipynb
- Added example-signal analyses (`134bdcc`): pick one NAc DA (dLight), PL (GCaMP), and
  NAc ACh (rAch) series via curated `intended_measurement`; full-session + 60 s traces,
  peri-go-cue averages, z-scored motion energy, FIP↔ME onset alignment, ME×FIP xcorr.
- Refactored into 4 phases (`e195cc0`): imports → data loading → data processing → data viz
  (42→37 cells). Removed old single-FIBER plots + FIBER/VARIANT scaffolding; consolidated
  helpers; excluded `pearsonR` series (signal-signal correlations, not photometry); curation
  set to `..._firstpass` (has `correct_mapping`; `secondpass` does not).
- Onset detection made causal: dropped the centered `uniform_filter1d` smoothing (acausal,
  biased onsets early). `threshold_onsets` now uses a sustained-crossing rule (`min_run`
  consecutive samples above threshold; onset time = true first crossing). `me_z` is raw
  z-scored ME (no pre-smoothing); the xcorr runs on raw z-scored traces with NaN-safe
  interpolation only.
- Motion energy: pad a leading 0 so ME is 1-to-1 with video frames. `aind-motion-energy`
  emits a consecutive-frame difference (N frames → N−1 values, no value for frame 0); the
  pad is gated on the ME metadata (`n_me_frames` vs `n_frames_decoded`) so it auto-disables
  once the library pads upstream. Length-mismatch warning now fires only on genuine anomalies.
- Simplified helpers: assert (don't sort) that `df_fip` timestamps are time-ordered per
  event after session-pick, so `get_trace` no longer re-sorts; merged
  `peri_event`/`peri_event_series` into one array-based
  `peri_event(t, y, event_times, censor=...)` (FIP traces pass `*get_trace(df_fip, ev)`).

### eph_00_single_unit_inspection.ipynb
- Import fix (`9f4b4ec`): `load_intermediate_data` / `find_session_dir` now come from
  `aind_dynamic_foraging_behavior_video_analysis.ephys.tongue_ephys`, not `data_loading`
  (where they don't exist — the old import raised ImportError).

### Other
- Started this CHANGELOG; CLAUDE.md Git-workflow section now points at it.
- Updated CLAUDE.md fip_00 description.
- Diagnosed bad curation JSON: `DA_NE_4channel_datacuration_secondpass.json` is malformed
  (line 4) and lacks `correct_mapping`; use `firstpass`.
