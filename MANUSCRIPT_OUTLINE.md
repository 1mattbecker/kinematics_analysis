# Manuscript outline — tongue kinematics in the dynamic foraging task

_Revision 2, 2026-09-17, against `wild` at `fb794c3` (merged into this branch). Revision 1 was
2026-09-14 against `bda2bc6`. What changed in the repo between the two: the five HOLD notebooks
were fully ported and archived (`kin_06`, `kin_07`, `eph_07`, `eph_09`, `spatial_axes.py`), the
lickometer validation grew from one notebook into the `val_02`–`val_05` series plus
`lickometer_qc.py`, and `REORG.md` was retired in favour of the map in `CLAUDE.md`. `kin_00`–`kin_05`
are unchanged. Numbers are from the pooled parquet (44 sessions) unless marked otherwise._

Status tags: **done** (pooled, executed, numbers in the repo) · **run, not captured** (executed
on Code Ocean but the committed notebook holds only the local skip-run) · **partial**
(single-session or unquantified) · **planned**.

---

## 0. The story in one paragraph

Lick-based tasks are the workhorse of head-fixed mouse neuroscience, and behavior in them is
almost always read out by a lickometer: a binary contact event that yields reaction time, choice,
and lick rate. High-speed video of the tongue shows the lickometer sees less than two-thirds of
what the tongue does, and the lickometer itself is not the problem: calibrated against each
session's own confirmed contacts, it misses under 1% of contact-like tongue excursions in the
median session. In a dynamic foraging task, ~39% of tongue movements never touch a spout, about
half of all trials contain at least one non-contact movement *before* the response lick, and
lickometer reaction time is therefore a composite: a first-movement latency plus an integer number
of ~170 ms sub-movement intervals. These preparatory movements are not noise. They are shorter and
slower than licks, they stop short of the spouts, and their direction predicts the upcoming choice
before the lickometer registers it (AUC 0.84 across held-out sessions, 0.94 within session), with
the cue-response lick then carrying the tongue further along the same direction on 84% of trials.
This reframes "reaction time" and "preparatory activity" in lick tasks and provides a kinematic
vigor readout for the LC-NE / catecholamine program.

---

## 1. Working titles

1. Covert tongue movements decompose reaction time and predict choice in a mouse foraging task
2. What the lickometer misses: tongue kinematics of decision execution in head-fixed mice
3. Kinematic structure of lick-based decisions in mice

## 2. Abstract skeleton

- Background: lick tasks + lickometer readout; the tongue is a reach-like effector (Bollu 2021).
- Approach: 500 Hz bottom-view video, Lightning Pose (11 keypoints, tongue-tip error ≈3.7 px),
  movement segmentation, alignment to lickometer and trial structure; 53 processed sessions from
  17 mice, of which 15 mice / 44 sessions / ~14.7k trials / ~246k movements pass the tongue
  quality filter.
- Result 1 (**done**): lickometer vs video. Video contact detection agrees with the lickometer on
  contacts (F1 0.93–0.95); the lickometer misses a median 0.8% of contact-like excursions; yet
  38.5% of tongue movements make no contact and are kinematically distinct.
- Result 2 (**done**): RT decomposition. The cue-response lick is the first movement on only
  ~50% of trials; lick latency grows stepwise with movement ordinal k (session means
  0.26 → 0.36 → 0.51 → 0.66 s); RT ≈ RT₁ + (k−1)·Δt with Δt ≈ 172 ms; within-k spread does not
  grow.
- Result 3 (**done**): preparatory movements carry choice. Last pre-lick movement direction
  decodes L/R at AUC 0.84 across held-out sessions (null 0.53) and median 0.94 within session;
  direction, not vigor, carries it; the confound with perseveration (P(stay) = 0.91) is
  quantified, not yet removed.
- Result 4 (**run, not captured / partial**): value, outcome, and session state vs vigor. Fill
  after the `kin_07` Code Ocean outputs are captured and `kin_04` §6 is quantified.
- Significance: RT and "preparatory" signals in lick tasks include overt covert movement; a
  validated multidimensional vigor readout for neuromodulator studies.

---

## 3. Introduction (paragraph plan)

1. **Lick tasks and the lickometer.** Directional licking is the dominant head-fixed readout
   (Guo et al. 2014 procedures; ALM preparatory-activity literature, Li et al. 2015/2016,
   Inagaki et al. 2019; dynamic foraging, Bari et al. 2019, Hattori et al. 2019; AIND task).
   RT, choice, and lick rate are derived from contact events.
2. **The tongue as a reach.** Kilohertz video shows licks are reach-like with corrective
   submovements controlled by ALM and superior colliculus (Bollu et al. 2021 Nature; Xu et al.
   2022 Nature; collicular map, Nature 2024; interception preprint 2026). Those studies use
   drinking or target-tracking with one goal and no value learning. **Gap:** no account of
   tongue kinematics inside a value-based choice task across many sessions.
3. **Vigor and value.** Reward and RPE modulate saccade and reach vigor (Shadmehr lab; reach
   vigor reflects RPE, 2025); dopamine and vigor (Hughes et al. 2020; Hamilos et al. 2021 eLife;
   counterpoint: subsecond DA does not specify vigor, Nat Neurosci 2025). LC-NE neurons encode
   RT and movement vigor in this task (Su et al., companion LC paper). **Gap:** no trial-level
   test of whether tongue vigor tracks model-derived value/RPE, and no kinematic definition of
   vigor that separates latency, number of sub-movements, speed, and extent.
4. **This paper.** Four findings, one dataset, open pipeline (AIND video-analysis package).

---

## 4. Results and figure plan

### Figure 1 — Task, video, and a movement-level description of the tongue  **[partial]**
- a. Dynamic foraging task schematic (go cue, L/R spouts, block-wise reward probabilities).
- b. Bottom-camera frame with the 11 Lightning Pose keypoints (`pixel_error.ipynb`). Note the
  bottom camera mirrors left/right: the keypoint named `spout_l` is the animal's right spout
  (`kin_06` §3).
- c. Processing chain on one trial: raw → confidence-masked (0.90) → 50 Hz low-pass → segmented
  movements (`model_quality.ipynb` walkthrough; library `kinematics_filter`,
  `segment_movements_trimnans`).
- d. Example trial: tongue y(t) with go cue, lickometer contacts, segments colored lick / non-lick
  (`kin_02` §11, Code Ocean only).
- e. Keypoint accuracy: test-set pixel error per keypoint (tongue tip center 3.67 ± 2.0 px);
  likelihood vs error (`pixel_error.ipynb`).
- f. Session inclusion (coverage > 90%, median movement duration > 60 ms;
  `test_session_quality_analysis`, `build_all_tongue_movements.py`) and the dataset table:
  53 processed sessions / 17 mice → 44 sessions / 15 mice pass; 14,727 trials; 246,359 movements
  (2024-05 → 2025-07).
- **To do:** pipeline schematic; per-mouse session counts; decide the movement-level noise
  criterion (decision 1) and report what it removes; the anatomical scale (jaw→spout ≈ 69 / 78 px,
  spout-to-spout ≈ 137 px in the example session, `kin_06` §3) belongs here or in Methods.

### Figure 2 — Two detectors of one behavior: lickometer vs video  **[done]**
- a. Video contact detection (tongue within 30–35 px of a spout, 100 ms refractory) vs
  lickometer, one session (`val_02`): at 30 px / 0.1 s / 0.1 s, 5,263 matched, 3.6% pose-only,
  10% lickometer-only, F1 0.93; at 35 px, 5.0% / 5.3%, F1 0.95. Distance from tongue tip to spout
  at a lickometer lick: median 11 px, 90th pct 30 px; only 1 of 5,846 lickometer licks had no
  tracked tongue within ±100 ms (`val_03` §4). Precision/recall vs threshold: 0.96 / 0.90 at
  30 px, 0.94 / 0.94 at 35 px (`val_03` §5).
- b. **Lickometer miss rate across 53 sessions** (`val_04`, `lickometer_qc.py`): "contact-like"
  calibrated per session against that session's own lickometer-confirmed contacts (depth ≥ median
  confirmed approach, dwell within the confirmed range); candidate = contact-like, no lickometer
  event within 100 ms of closest approach, lickometer active. 36,951 confirmed vs 596 candidates;
  median session miss rate 0.8% (range 0–33%); 49 / 53 sessions pass the gates; 3 sessions
  flagged (791691 ×2 at ~10%, 751004 2024-12-22 at 8%), all with the intermittent-fault
  day-to-day pattern. The lickometer fires a median 17 ms before closest approach (range −66 to
  +5 ms across sessions). Ranking is stable to every parameter except a 50 ms window (`val_05` §6).
- c. Three-way correspondence, pooled (`kin_05` §3): movements without licks 38.5%
  (94,955 / 246,359); lick-movements with > 1 contact 5.7% (per-session median 2.9%); licks
  without a movement 0.26% (single session; pooled version still needs per-session lick tables).
- d. Kinematic profile, lick vs non-lick: duration median 86 vs 50 ms; total distance mean 112 vs
  54 px; peak velocity median 3,652 vs 2,243 px/s (`kin_05` §6, `kin_01` §5); max excursion from
  jaw 65 ± 11 vs 41 ± 16 px, per-session paired Wilcoxon p = 1e-13 (`kin_06` §5, jaw-centred
  2-D density).
- e. Per-trial structure: median 12 movements/trial, 7 lick + 3 non-lick, 29% non-lick;
  lick vs non-lick counts per trial r = 0.69 (`kin_05` §7).
- **Message (strengthened since rev 1):** non-contact movements are a large, kinematically
  distinct class, and they are not lickometer misses: the lickometer's own miss rate is ~1%.
- **Cohort note for the text:** `val_*` runs on all 53 processed sessions; `kin_*` on the 44 that
  pass the tongue quality filter. State both.
- **Methods check surfaced by `val_05`:** the library matches lick frames to movements with a
  10 ms tolerance (`annotate_licks_in_kinematics`), while the lickometer leads closest approach
  by 10–30 ms. Movement-level `has_lick` is unaffected (the contact frame falls inside an 86 ms
  movement), but confirm the frame-level `lick` flag is not systematically placed on the approach
  phase before any within-movement timing is reported.

### Figure 3 — Population kinematics of a tongue movement  **[partial]**
- a. Distributions of duration, peak velocity, total distance, excursion angle, outbound
  duration/velocity (`kin_01` §4).
- b. Speed–amplitude scaling (peak velocity vs distance), the lick "main sequence"
  (`kin_01` §7); compare to Bollu 2021 lick kinematics.
- c. Session-level consistency across 44 sessions (median duration 0.03–0.07 s, median peak
  velocity 1.4–3.5 k px/s) (`kin_01` §6).
- d. UMAP of movement features: discrete clusters or a continuum; feature–axis correlations;
  one session overlaid (`kin_03`). **Result still not interpreted.**
- e. Movement-onset-aligned mean speed profile and within-movement time to contact. **Not
  ported** — lives only in archived `tongue_kinematics_cueresponse` cells 112–113 (see §7 below).
- f. Feature redundancy, which any later screen must respect: among the 24 kinematic columns,
  21 pairs exceed |ρ| = 0.7 and five are identical once session offsets are removed
  (`max_x` = `max_x_from_jaw` = `max_x_distance`; `time_to_endpoint` = `out_duration`;
  `endpoint_y` ≈ `max_y_from_jaw`) (`kin_07` §6.1). Use this to pick 4–5 canonical features and
  reuse them in every later figure.
- **Column-name trap for the text and methods:** `endpoint_x/y` and `max_*_from_jaw` are
  absolute camera pixels; the jaw-relative distances are `max_x_distance` / `max_y_distance`.
  Pooled anatomical claims must use the jaw-centred frame from `kin_06` §2.1 (real jaw keypoint
  on Code Ocean, algebraic fallback locally, the two agree to 0.11 px).

### Figure 4 — Reaction time decomposes into first-movement latency plus sub-movements  **[done]**
- a. Example session rasters: movements per trial at latency from go, colored by type
  (cue-response lick / other lick / non-lick) and by ordinal (`kin_02` §11).
- b. Cue-response movement ordinal k across 14,138 trials: k = 1 49.5%, k = 2 39.0%,
  k = 3 8.8%, k ≥ 4 2.7%.
- c. Lick latency by k: session-mean ± SEM 0.258, 0.358, 0.512, 0.660 s (n = 44, 44, 43, 38
  sessions); trial medians 0.17, 0.31, 0.51, 0.70 s (`kin_02` §4, §10).
- d. Video first-movement latency (k = 1 median 0.132 s) vs lickometer RT: different quantities
  on different event streams (`kin_02` §8 prose).
- e. Δt = median within-trial inter-movement interval = 0.172 s (≈ 5.8 Hz, the lick rhythm);
  de-shifting lick latency by (k−1)·Δt largely superimposes k = 2–4 on k = 1; KS tests still
  reject full collapse (`kin_02` §8–§9).
- f. Spread does not accumulate: SD of lick latency 0.12–0.145 s at every k (bootstrap CI), so
  preparatory sub-movements add a deterministic offset, not noise (`kin_02` §10).
- g. k = 1 latency is right-skewed, approximately log-normal (fit + QQ; report effect size, not
  p, at n ≈ 7k) (`kin_02` §5; the ephys-side version is `eph_01` §8b).
- h. Session-level: does median k = 1 latency predict median k = 2 latency? (`kin_02` §7; result
  not yet recorded).
- **Message:** lickometer RT = RT₁ + (k−1)·Δt. The residual mismatch at k = 1/2 (secondary bump)
  points to further covert movements → Figure 5.

### Figure 5 — Preparatory movements carry choice information  **[done, pooled]**
_All of this is now `kin_06`, executed on Code Ocean with real jaw keypoints for 44 / 44 sessions._
- a. Landmark frame: jaw, left/right spout, cue-response lick endpoints by side
  (`kin_06` §3–§4, one session, Code Ocean only). Pooled equivalent: cue-response `endpoint_y`
  in the jaw frame, −60 ± 15 px (left) vs +44 ± 14 px (right).
- b. Prevalence: 50.4% of trials have ≥ 1 non-lick movement before the cue-response lick,
  11.5% have ≥ 2; mean 1.30 (`kin_05` §8).
- c. Non-lick endpoints by ordinal: the first movement of a trial sits near the midline and
  short (x 19 px, y −15 px), ordinals 3–4 are shortest (x 6–7 px), later ones drift toward a
  spout (`kin_06` §6).
- d. Last pre-lick movement direction by subsequent lick side: excursion angle median −60°
  (left) vs +38° (right), per-session Wilcoxon p = 1e-11; binned P(right lick) vs jaw-centred
  `endpoint_y` runs 0.08 → 0.94 and vs angle 0.09 → 0.88, session as the unit (`kin_06` §7).
- e. Ridge-logistic decode of L/R from last-pre-lick kinematics, 7,077 trials × 8 features
  (`kin_06` §8): `GroupKFold` by session AUC **0.835 ± 0.10** (pooled out-of-fold 0.836,
  accuracy 0.81); within-session shuffled-label null 0.53, p = 0.005; per-session fits median AUC
  **0.943** (39 sessions with ≥ 40 trials; 97% above 0.5). Permutation importance: `last_angle`
  0.17, `mean_distance` 0.10, `mean_angle` 0.07; both peak-velocity terms and `session_time` ≈ 0.
  **Direction, not vigor, carries the decode.**
- f. Perseveration confound, quantified (`kin_06` §8.4): P(stay) = 0.91; previous choice alone
  AUC 0.907; kinematics + previous choice 0.954; kinematics *within* previous-choice strata
  0.815 / 0.772. Not separated here — the block-aware version is the top open analysis (§8, item 2).
- g. Pre-lick → cue-response endpoint displacement (`kin_06` §9): the lick carries the tongue
  further from the midline on the same side (left +23 px, right +15 px); same-side fraction
  0.84 ± 0.02 (session unit); per-session r(pre-lick y, cue-response y) median 0.78.
- h. **Changed-mind trials** (new since rev 1; `kin_06` §9): 1,185 of 7,138 trials (17%) the
  pre-lick pointed at the other spout. Their pre-lick excursions are shallower (29 vs 32 px) and
  the switch fraction ranges 3–43% across the 15 animals, with a matching per-animal bias in
  starting side (6–76% start right). Candidate panel: a behavioral "vacillation" index.
- **Message:** the choice is embodied in overt tongue motion before contact. Implication for
  "preparatory" neural activity in lick tasks (Discussion 2).

### Figure 6 — Outcome and value modulate tongue vigor  **[run, not captured / partial]**
- a. First-movement vigor (peak velocity, latency) on trials after reward vs after no reward,
  within-session z-scored (`kin_04` §6; built, **result not yet quantified**).
- b. Trajectories and endpoints coloured by `q_diff` / `q_diff_c` in the landmark frame
  (`kin_07` §4; one session, Code Ocean).
- c. The four source-pinned pairs as binned scatter + per-session slope, Wilcoxon across
  sessions, FDR over 4 (`kin_07` §5) — descriptive only; the pairs were chosen after looking.
- d. Systematic screen, 24 kinematic × 11 model latents, four ways: Spearman (session unit,
  FDR over the whole grid), mutual information (trial, null-corrected), RidgeCV held-out R² with a
  20-shuffle permutation p (the only genuinely predictive panel; a latent counts as predicted
  only if R² > 0 and p_perm < 0.05), random-forest importance (`kin_07` §6).
- e. Previous-trial RPE vs current-trial kinematics, on the cue-response movement and on the
  trial mean, with same-trial RPE as the drift reference (`kin_07` §7).
- **Status:** `kin_07` ran end-to-end on Code Ocean on 2026-09-16 and the user reported it good,
  but the committed notebook is the local skip-run and no numbers are recorded in `TODO.md` or
  `CHANGELOG.md`. Coverage: 40 / 44 sessions have a `QLearning_L2F1_CKfull_softmax` fit; the four
  751004 sessions have none. **First action for this figure: capture the Code Ocean outputs
  (commit the executed notebook or record the §3.5 L/R check, the starred RidgeCV latents, §6.3,
  and §7).**
- **Design notes already settled in `kin_07`:** `q_diff` = R − L (the two archived copies
  disagreed in sign); one row per trial, session as the sampling unit (the sources pseudo-
  replicated at movement level); `q_sum` added as the vigor-like control.
- **Known limitations (`kin_07` §8):** the screen conditions on nothing (value covaries with
  block, choice side, time); `excursion_angle_deg` treated as linear; one agent, no model
  comparison; sessions treated as exchangeable across 13–15 subjects.

### Figure 7 — Slow dynamics of vigor across a session  **[partial]**
- a. Single-session RT vs trial with 50/100-trial moving average.
- b. Population: z-scored RT → 100-trial MA → 0–1 session fraction → grand mean ± SEM.
- c. Vigor timecourse, first-movement vs all-movements; which features drift most (slopes).
- d. Relation to engagement/satiety proxies (response rate, model variables). `val_05` §2 shows
  what disengagement looks like in the raw streams (lickometer rate collapses while short tongue
  excursions continue in the final half hour), a useful qualitative anchor.
- **Option:** fold into Figure 6 as "state" panels if effects are modest. `kin_04` is unchanged
  since rev 1; nothing here is quantified yet.

### Optional Figure 8 — LC single-unit encoding of tongue kinematics  **[eph_* series]**
- RT and kinematic encoding in LC units (`eph_01`–`eph_03`), spatial organization (`eph_04`),
  timing (`eph_05`), structural axes (`eph_09`, ran on Code Ocean incl. MERFISH; `eph_08`
  reproduces the poster figure: r = 0.184, p = 0.068, n = 99).
- Within-trial vs ITI movement bouts (`eph_07`): written and locally verified for the bout
  helpers (246,359 movements → 31,081 bouts; 14,368 go-responsive / 9,297 ITI), **but §5 onward
  has not run on Code Ocean**. The only numbers (103 units; go-responsive bout Δ +3.2 Hz vs ITI
  +0.1 Hz) sit in the stored outputs of the archived
  `tongue_kinematics_ephys_intertrialmovs.ipynb` — do not strip them.
- **Recommendation unchanged:** keep this paper behavioral and cite the LC paper; if a neural
  hook is wanted, run `eph_07` on Code Ocean first.

### Supplementary figures
- S1 Noise criteria (`kin_00`): 10-frame duration floor (9.7% of detections), single-frame 2.3%,
  spatial outliers r_max_norm > 4×: 0.5%; the 2-frame velocity–distance degeneracy; video
  ground-truth frames. `val_05` §2–§3 now supplies the mechanism for the short detections: a
  keypoint jittering at the spout crosses a 10 px threshold 11–22 times per second, and
  one-frame excursions are unconfirmed by the lickometer 13% of the time vs 1–2% for 9–40 frames.
- S2 Filter-threshold sensitivity curves (`kin_01` §3b).
- S3 Lickometer validation methods (`val_02` sweep heatmaps and ILI distributions; `val_05`
  depth/dwell tables, onset-vs-closest-approach, window sensitivity, greedy-vs-optimal matching).
- S4 Multi-lick movements and licks-without-movements examples (`kin_05` §4–§5).
- S5 Per-session versions of Figs 4c, 4e; KS table; bootstrap SD CIs.
- S6 Per-session decode AUCs and permutation importances; changed-mind and switch-fraction
  tables by animal (`kin_06` §8–§9).
- S7 Value-screen matrices and the kinematic collinearity matrix (`kin_07` §6.1, §6).

---

## 5. Discussion (points to make)

1. **RT in lick tasks is a composite.** The video first-movement latency and the lickometer RT
   differ by an integer number of ~170 ms sub-movement intervals. Studies that regress neural
   activity on lickometer RT (including the companion LC paper, which defines RT as first tongue
   detection with a 1 s cutoff) are mixing latency with sub-movement count. Propose reporting
   RT₁ and k separately.
2. **"Preparatory" activity may be movement.** Pre-lick tongue kinematics predict choice at
   AUC 0.84–0.94; any preparatory-period neural signal in a lick task must be checked against
   overt tongue motion (cf. uninstructed-movement literature, Musall et al. 2019, Stringer et al.
   2019; ALM preparatory activity). State plainly that perseveration predicts choice even better
   (AUC 0.91) and that kinematics survive within previous-choice strata.
3. **Non-contact movements are a behavioral class.** Shorter, slower, jaw-proximal; prevalent
   in ~50% of trials before the response; not lickometer misses (median 0.8%). Their count
   correlates with lick count per trial (r = 0.69), suggesting shared engagement drive; their
   direction sometimes disagrees with the final lick (17% of trials), a vacillation signature that
   varies 3–43% across animals.
4. **Vigor is multidimensional.** Latency, sub-movement count, peak velocity, and extent
   dissociate (Fig 4f: spread constant across k; Fig 5e: direction carries choice, velocity does
   not). This is the kinematic vocabulary the neuromodulator program needs (LC-NE → RT; DA →
   velocity, per the grant framing). Say what `kin_07` finds about value once captured.
5. **Relation to the tongue-control literature.** Sub-movements here are preparatory and
   choice-bearing, not target-correction (Bollu 2021); 2D bottom view vs 3D; decision task vs
   drinking.
6. **Limitations.** 2D tracking; lickometer as imperfect reference with no hand-scored ground
   truth yet (`val_04` §7: no candidate missed lick has been scored by eye); head-fixed; movement
   segmentation depends on confidence masking (val series runs at 0.8, pipeline at 0.9); noise
   criterion choice; pixel scale varies 1.8× across sessions.

---

## 6. Methods (sections, with the numbers already fixed)

- Animals, task, sessions: 17 mice / 53 processed sessions; 15 / 44 after the tongue quality
  filter; training stage; ephys vs non-ephys sessions.
- Video: FLIR Blackfly S, 500 Hz, 720 × 540, bottom view (mirrors L/R); camera–behavior clock
  alignment (`video_alignment`); video starts ~7 min before the first go cue.
- Pose estimation: Lightning Pose, 11 keypoints, ~1.4k labeled frames; test pixel error per
  keypoint; likelihood threshold 0.90.
- Kinematic processing: cubic interpolation of masked gaps, 4th-order 50 Hz zero-phase
  Butterworth, velocity by differentiation; segmentation with ≤ 3 dropped frames; outbound
  phase = onset → max excursion.
- Movement-level features (the 49 columns) and the jaw-centred frame (`kin_06` §2.1).
- Lick alignment: contact ↔ frame within 10 ms; cue-response movement; ordinal k;
  `movement_before_cue_response`; the two latency definitions; lickometer leads closest
  approach by ~17 ms.
- Session inclusion and the movement-level noise criterion.
- Video contact detection and lickometer QC (`val_02`–`val_05`): spatial threshold, refractory
  100 ms, match window 100 ms, per-session contact-like calibration, gates and flags.
- RT decomposition: Δt estimate, de-shift, KS, bootstrap SD.
- Choice decoding: standardized ridge-logistic, `GroupKFold` by session, within-session
  shuffled-label null, per-session fits, permutation importance on held-out sessions.
- Behavioral model latents: MLE fits (`QLearning_L2F1_CKfull_softmax`), derived `q_diff`,
  `q_sum`, `q_diff_c`, `chosen_prob`, L/R convention checks, one row per trial, session as the
  sampling unit, BH-FDR over the full screen grid, RidgeCV permutation p.
- Statistics: session as sampling unit for population claims; effect sizes with CIs; note the
  Wilcoxon p-value floor at n = 44.
- Code/data availability: `aind-dynamic-foraging-behavior-video-analysis`, this capsule.

---

## 7. Archive audit — what has not been ported into the `kin_*` series

The five HOLD notebooks were archived on 2026-09-16 with every gate in `TODO.md` met. A cell-level
pass over the archived notebooks against `kin_00`–`kin_07`, `val_02`–`val_05`, `model_quality`, and
`eph_01`/`eph_07` finds the port complete for everything the plan tracked. Five items sat outside
the plan's cell ranges; none blocks the paper, and only the first is worth porting.

| Archived source | Content | Where it is now | Verdict |
|---|---|---|---|
| `tongue_kinematics_cueresponse` cells 112–113 | Movement-onset-aligned mean speed profile (±50 ms, aligned to onset or peak x-velocity), within-movement time from onset to lickometer contact, initial speed at frames 0/1 | **Nowhere** (no `kin_*` uses per-frame speed profiles) | **Port** into `kin_01` as a new section (Code Ocean, per-frame `tongue_kins`). It is the Fig 3e "what a lick looks like" panel and gives the onset-to-contact latency the Methods need |
| `tongue_kinematics_cueresponse` cells 63–66 | 1-D histograms of excursion angle (polar and linear) and peak velocity, all vs non-lick | Angle only as the 2-D jaw-frame density (`kin_06` §5); velocity in `kin_01` §5 / `kin_05` §6 | Optional: add a 1-D angle histogram lick vs non-lick to `kin_06` §5 |
| `tongue_kinematics_cueresponse` cells 33–34 | `plot_reach_vectors` (start→end arrows coloured by a variable over landmarks) | Value-coloured trajectories in `kin_07` §4 | Skip; cosmetic variant |
| `tongue_kinematics` cell 76 | `plot_stacked_trials`: five consecutive trials' tongue traces, 3 s window, coloured by `has_lick` | Single-trial version in `kin_02` §11 (Code Ocean) | Optional multi-trial variant for Fig 4a |
| `tongue_kinematics` cell 93 | Cue-response peak velocity on trial n vs n+1 coloured by reward | Intent covered by `kin_04` §6 | Skip |

Everything else outside the planned ranges is either library code that now lives in the package
(`tongue_kinematics` 24–33, 102–106), pipeline QC plots now in `model_quality.ipynb`
(`tongue_kinematics` 5–23), superseded latency panels (`tongue_latency` 10–13, 27–34 →
`kin_02` §5/§11 and `eph_01` §8b), or commented-out scratch.

Two things in the archive are worth protecting rather than porting: the stored outputs of
`spatial_axis_comparison_rt_encoding.ipynb` (already flagged in `code/archive/README.md`) and of
`tongue_kinematics_ephys_intertrialmovs.ipynb`, which holds the only executed numbers for the
within-trial vs ITI bout comparison until `eph_07` runs on Code Ocean.

---

## 8. Open decisions and checks before drafting

1. **Movement-level noise criterion.** `kin_01`'s `out_duration > 0.05 s` removes 61% of
   detections; `kin_00`'s physically grounded criterion removes 10%. Recommend `kin_00`'s and
   report sensitivity in S2. `val_05` §3 now gives the mechanism for the short detections, which
   strengthens the case for the 10-frame floor.
2. **Block-aware conditioning.** Both `kin_06` §8.5 and `kin_07` §8 defer it. Choice, value,
   and time-in-session covary; previous choice alone reaches AUC 0.91. The reviewer-proof version
   of Figs 5 and 6 conditions the decode and the screen on previous choice / block (partial
   correlation or within-stratum), and this is the one analysis still missing for the two headline
   claims.
3. **Capture `kin_07`'s Code Ocean outputs.** The run happened; the numbers are not in the
   repo. Until they are, Figure 6 has no result to state.
4. **RT definition alignment with the LC paper.** Report both `movement_latency_from_go` and
   `lick_latency`, state which the LC paper used, and show how k affects each.
5. **Ephys in or out.** Behavioral paper + cite LC paper (recommended), or add Figure 8 after
   `eph_07` runs.
6. **Ground truth for the lickometer miss rate.** No `val_04` candidate has been scored by eye.
   A hand-scored sample (~50 candidates from the flagged sessions, clips are already generated)
   would let Fig 2b claim a miss *rate* rather than a candidate rate.
7. **Frame-level lick tolerance** (10 ms) vs the measured lickometer lead (~17 ms): confirm
   the frame flag lands inside the contact, not the approach.
8. **Mouse metadata**: sex, age, training history, which sessions carried ephys, which have model
   fits (40 / 44).

## 9. Suggested order of remaining analyses (value ÷ effort)

1. Capture `kin_07` outputs from the Code Ocean run → Figure 6b–e.
2. `kin_04` §6 post-reward vigor (pooled parquet only) → Figure 6a.
3. Block-aware decode and screen (extend `kin_06` §8.4 and `kin_07` §6) → the version of
   Figs 5e and 6d that goes in the abstract.
4. `kin_03` UMAP conclusion → Figure 3d.
5. Port `cueresponse` 112–113 (speed profile, onset-to-contact) into `kin_01` → Figure 3e.
6. `kin_04` session timecourse quantified → Figure 7.
7. `kin_00` criterion decision + rerun `kin_01`/`kin_02`/`kin_05`/`kin_06` under it.
8. Hand-score a sample of `val_04` candidates → Figure 2b.
9. Pooled licks-without-movements (`kin_05` §4) → Figure 2c completion.
10. `eph_07` on Code Ocean, only if Figure 8 is in.

## 10. Literature gaps this work fills

| Existing work | What it established | Gap your data fills |
|---|---|---|
| Bollu et al. 2021 *Nature*; Xu et al. 2022 *Nature*; collicular map 2024 *Nature*; interception preprint 2026 | Licks are reach-like with corrective submovements; ALM/SC control tongue direction | Tongue kinematics inside a value-based *choice* task, over 44 sessions and 246k movements; sub-movements that are preparatory/choice-bearing, not corrective |
| Lickometer device papers (J Neurophysiol 2019; eNeuro 2024 optical; capacitive 2025) | Validate contact detectors against video | Quantify what contact detection *cannot* see (39% of tongue movements, ~50% of trials with pre-response movement) and, separately, how rarely it misses a contact (median 0.8%) with a per-session calibrated QC |
| RT literature in mice (motor-readiness 2023; "humanlike RT" preprint 2024; LATER/log-normal models) | RT distributions and readiness | RT = RT₁ + (k−1)·Δt; constant within-k spread; video vs lickometer RT are different variables |
| ALM preparatory activity (Li 2015/2016; Inagaki 2019); uninstructed movements (Musall 2019; Stringer 2019) | Neural choice signals before the lick; movements explain neural variance | Overt tongue kinematics predict choice before contact (AUC 0.84 held-out, 0.94 within session) — a behavioral confound/mechanism for "preparatory" signals |
| Vigor & value (saccade/reach vigor and RPE; Hughes 2020; Hamilos 2021; Nat Neurosci 2025) | Reward modulates vigor of limb/eye movements; DA links debated | Trial-level tongue vigor vs model-derived Q/RPE and post-outcome state in a foraging task; a multidimensional vigor definition |
| Companion LC paper (Su et al.) | LC-NE neurons encode RT and vigor; RT defined from first tongue detection | Behavioral foundation and refinement of the RT variable those neural analyses use |

## 11. Key references to collect

- Bollu T. et al. Cortex-dependent corrections as the tongue reaches for and misses targets.
  *Nature* 2021.
- Xu D. et al. Cortical processing of flexible and context-dependent sensorimotor sequences.
  *Nature* 2022.
- A collicular map for touch-guided tongue control. *Nature* 2024.
- Precision in motion: reactive and anticipatory control of mouse tongue movement for
  interception. *bioRxiv* 2026.
- Guo Z.V. et al. Procedures for behavioral experiments in head-fixed mice. *PLoS One* 2014.
- Li N. et al. 2015 *Nature*; Li N. et al. 2016 *Nature*; Inagaki H. et al. 2019 *Nature* (ALM).
- Bari B.A. et al. Stable representations of decision variables for flexible behavior.
  *Neuron* 2019; Hattori R. et al. *Cell* 2019.
- Musall S. et al. *Nat Neurosci* 2019; Stringer C. et al. *Science* 2019.
- Hughes R.N. et al. Ventral tegmental dopamine neurons control the impulse vector during
  motivated behavior. *Curr Biol* 2020.
- Hamilos A.E. et al. Slowly evolving dopaminergic activity modulates the moment-to-moment
  probability of reward-related self-timed movements. *eLife* 2021.
- Subsecond dopamine fluctuations do not specify the vigor of ongoing actions. *Nat Neurosci* 2025.
- Rapid dopaminergic signatures in movement: reach vigor reflects RPE. 2025.
- Reward prediction error modulates saccade vigor (Shadmehr lab).
- Behavioral measurements of motor readiness in mice. 2023.
- Carpenter R.H.S. & Williams M.L.L. *Nature* 1995 (LATER); Noorani & Carpenter 2016.
- Biderman D. et al. Lightning Pose. *Nat Methods* 2024.
- Lickometer devices: *J Neurophysiol* 2019; *eNeuro* 2024 (optical); capacitive system 2025.
- Su Z. et al. Topographic structure and function of locus coeruleus norepinephrine neurons
  (companion).
