# Manuscript outline — tongue kinematics in the dynamic foraging task

_Drafted 2026-09-14 from the `kin_*`, HOLD, and validation notebooks on branch `wild`. Numbers are
from the pooled parquet `all_tongue_movements_04022026.parquet` (44 sessions) unless marked
"single session". Status tags: **done** (pooled, executed), **partial** (exists but single-session
or unquantified), **planned** (a `TODO.md` port or a new analysis)._

---

## 0. The story in one paragraph

Lick-based tasks are the workhorse of head-fixed mouse neuroscience, and behavior in them is
almost always read out by a lickometer: a binary contact event that yields reaction time, choice,
and lick rate. High-speed video of the tongue shows that the lickometer sees less than two-thirds
of what the tongue does. In a dynamic foraging task, ~39% of tongue movements never touch a spout,
about half of all trials contain at least one non-contact movement *before* the response lick, and
lickometer reaction time is therefore a composite: a first-movement latency plus an integer number
of ~170 ms sub-movement intervals. These preparatory movements are not noise: they are shorter and
slower than licks, they stop short of the spouts, and their direction predicts the upcoming choice
before the lickometer registers it. Together this reframes "reaction time" and "preparatory
activity" in lick tasks and provides a kinematic vigor readout for the LC-NE / catecholamine program.

---

## 1. Working titles

1. Covert tongue movements decompose reaction time and predict choice in a mouse foraging task
2. What the lickometer misses: tongue kinematics of decision execution in head-fixed mice
3. Kinematic structure of lick-based decisions in mice

## 2. Abstract skeleton

- Background: lick tasks + lickometer readout; tongue is a reach-like effector (Bollu 2021).
- Approach: 500 Hz bottom-view video, Lightning Pose (11 keypoints, tongue-tip error ≈3.7 px),
  movement segmentation, alignment to lickometer and trial structure; 15 mice, 44 sessions,
  ~14.7k trials, ~246k tongue movements.
- Result 1: lickometer vs video correspondence — video-based lick detection matches the lickometer
  (F1 0.95), but 38.5% of tongue movements have no contact; they are kinematically distinct.
- Result 2: RT decomposition — cue-response lick is the first movement on only ~50% of trials;
  lick latency grows stepwise with movement ordinal k (0.26 → 0.36 → 0.51 → 0.66 s, session means);
  RT ≈ RT₁ + (k−1)·Δt with Δt ≈ 172 ms; within-k spread does not grow with k.
- Result 3: preparatory movements carry choice — last pre-lick movement endpoint/angle predicts
  L/R lick (AUC 0.95 single session; pooled value TBD).
- Result 4: value/outcome and session-state modulation of vigor (post-reward vigor, Q/RPE
  screen, within-session drift) — **to be filled after kin_04 / kin_07 land**.
- Significance: RT and "preparatory" signals in lick tasks include overt covert movement;
  a validated kinematic vigor readout for neuromodulator studies.

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
- b. Bottom-camera frame with the 11 Lightning Pose keypoints (`pixel_error.ipynb`).
- c. Processing chain on one trial: raw → confidence-masked (0.90) → 50 Hz low-pass → segmented
  movements (`tongue_kinematics.ipynb` cells 9–11; library `kinematics_filter`,
  `segment_movements_trimnans`).
- d. Example trial: tongue y(t) with go cue, lickometer contacts, movement segments colored
  lick / non-lick (`kin_02` §11, Code Ocean only).
- e. Keypoint accuracy: test-set pixel error per keypoint (tongue tip center 3.67 ± 2.0 px);
  likelihood vs error (`pixel_error.ipynb`).
- f. Session inclusion (coverage > 90%, median movement duration > 60 ms;
  `test_session_quality_analysis`, `build_all_tongue_movements.py`) and dataset table:
  15 mice, 44 sessions (2024-05 → 2025-07), 14,727 trials, 246,359 movements.
- **To do:** clean pipeline schematic; per-mouse session counts; decide the movement-level
  noise criterion (see §8, decision 1) and report what it removes.

### Figure 2 — Two detectors of one behavior: lickometer vs video  **[done, one CO-only panel]**
- a. Video-based lick detection (spout proximity 30–35 px + 100 ms refractory) vs lickometer:
  parameter sweep and best operating point, recall 0.97 / precision 0.93 / F1 0.95
  (`tongue_lickometer.ipynb`). Establishes the two streams agree *on contacts*.
- b. Three-way correspondence: movements without licks 38.5% (94,955 / 246,359); lick-movements
  with > 1 contact 5.7% (per-session median 2.9%); licks without a movement 0.26% (single session;
  pooled version needs per-session `nwb_df_licks`, `kin_05` §3a/§4).
- c. Kinematic profile, lick vs non-lick: duration (median 86 vs 50 ms), total distance
  (mean 112 vs 54 px), peak velocity (median 3,652 vs 2,243 px/s), outbound metrics
  (`kin_05` §6, `kin_01` §5).
- d. Max excursion from jaw, lick vs non-lick, over jaw/spout landmarks: non-lick movements stop
  short of the spouts (`cueresponse` cell 95 == `tongue_kinematics` cell 60 → `kin_06` §5).
- e. Per-trial structure: median 12 movements/trial, 7 lick + 3 non-lick, 29% non-lick;
  lick vs non-lick counts per trial r = 0.69 (`kin_05` §7).
- **Message:** non-contact movements are a large, kinematically distinct class, not detector noise.

### Figure 3 — Population kinematics of a tongue movement  **[partial]**
- a. Distributions of duration, peak velocity, total distance, excursion angle, outbound
  duration/velocity (`kin_01` §4).
- b. Speed–amplitude scaling (peak velocity vs distance), the lick "main sequence"
  (`kin_01` §7); compare to Bollu 2021 lick kinematics.
- c. Session-level consistency across 44 sessions (median duration 0.03–0.07 s, median peak
  velocity 1.4–3.5 k px/s) (`kin_01` §6).
- d. UMAP of movement features: discrete clusters (lick / non-lick / grooming?) or a continuum;
  feature–axis correlations; one session overlaid (`kin_03`). **Result not yet interpreted.**
- e. Movement-onset-aligned mean speed profile (`cueresponse` cell 113 helpers; per-frame, CO).
- **To do:** run `kin_03` to a conclusion; pick 4–5 canonical features and use them everywhere.

### Figure 4 — Reaction time decomposes into first-movement latency plus sub-movements  **[done]**
- a. Example session rasters: movements per trial at latency from go, colored by type
  (cue-response lick / other lick / non-lick) and by ordinal (`kin_02` §11).
- b. Cue-response movement ordinal k across 14,138 trials: k = 1 49.5%, k = 2 39.0%,
  k = 3 8.8%, k ≥ 4 2.7%.
- c. Lick latency by k: session-mean ± SEM 0.258, 0.358, 0.512, 0.660 s (n = 44, 44, 43, 38
  sessions); trial medians 0.17, 0.31, 0.51, 0.70 s (`kin_02` §4, §10).
- d. Video first-movement latency (k = 1 median 0.132 s) vs lickometer RT: the two RTs are
  different quantities on different event streams (`kin_02` §8 prose).
- e. Δt = median within-trial inter-movement interval = 0.172 s (≈ 5.8 Hz, i.e. the lick
  rhythm); de-shifting lick latency by (k−1)·Δt largely superimposes k = 2–4 on k = 1;
  KS tests still reject full collapse (`kin_02` §8–§9).
- f. Spread does not accumulate: SD of lick latency 0.12–0.145 s at every k (bootstrap CI),
  so preparatory sub-movements add a deterministic offset, not noise (`kin_02` §10).
- g. k = 1 latency is right-skewed, approximately log-normal (fit + QQ; note the normality
  tests reject at n = 7k, report effect size not p) (`kin_02` §5).
- h. Session-level: does median k = 1 latency predict median k = 2 latency (arousal-state
  covariation)? (`kin_02` §7; result not yet recorded).
- **Message:** lickometer RT = RT₁ + (k−1)·Δt. Residual mismatch at k = 1/2 (secondary bump)
  points to further covert movements → Figure 5.

### Figure 5 — Preparatory movements carry choice information  **[partial → `kin_06`]**
- a. Landmark frame: jaw, left/right spout, cue-response lick endpoints by side
  (`cueresponse` cells 19, 93, 94; CO only for absolute landmarks).
- b. Prevalence: 50.4% of trials have ≥ 1 non-lick movement before the cue-response lick,
  11.5% have ≥ 2; mean 1.30 (`kin_05` §8, donut by count).
- c. Last pre-lick movement excursion angle and endpoint-y by subsequent lick side (violins);
  binned P(right lick) vs endpoint-y and vs angle (`cueresponse` cells 49, 55).
- d. Ridge-logistic decode of L/R from last pre-lick kinematics: single session AUC 0.95,
  accuracy 0.90 (n = 68 held-out trials); coefficient and permutation importance
  (`cueresponse` cells 44–48). **Pool across sessions, session as the sampling unit; report
  AUC distribution and chance via label shuffle.**
- e. Pre-lick → cue-response endpoint displacement per trial against the jaw midline
  (`cueresponse` cells 50–52).
- f. Non-lick endpoints by movement ordinal (do earlier movements aim less?) (`cueresponse`
  cells 54, 56).
- **Message:** the choice is embodied in overt tongue motion before contact. Implication for
  "preparatory" neural activity in lick tasks (see Discussion).

### Figure 6 — Outcome and value modulate tongue vigor  **[planned → `kin_04` §6, `kin_07`]**
- a. First-movement vigor (peak velocity, latency) on trials after reward vs after no reward,
  within-session z-scored (`kin_04` §6; built, result not yet quantified).
- b. Trajectories and endpoints colored by Q-value difference / chosen probability
  (`cueresponse` cells 18–26; single session).
- c. Binned kinematics vs value: q_diff × excursion, chosen_prob × extent (`cueresponse` cells
  21–28).
- d. Systematic screen, all kinematic × all model latents: Spearman ρ, mutual information,
  RidgeCV R², random-forest importance (`tongue_kinematics` cell 135).
- e. Previous-trial RPE vs current-trial kinematics (`tongue_kinematics` cell 137).
- **Dependencies:** `get_mle_model_fitting` (Han's pipeline) per session; pooling is the main
  cost. Land (a) first — it needs only the pooled parquet.

### Figure 7 — Slow dynamics of vigor across a session  **[partial → `kin_04`]**
- a. Single-session RT vs trial with 50/100-trial moving average.
- b. Population: z-scored RT → 100-trial MA → 0–1 session fraction → grand mean ± SEM.
- c. Vigor timecourse, first-movement vs all-movements; which features drift most (slopes).
- d. Relation to engagement/satiety proxies (response rate, model-derived variables).
- **Option:** fold into Figure 6 as "state" panels if effects are modest.

### Optional Figure 8 — LC single-unit encoding of tongue kinematics  **[exists in `eph_*`]**
- RT and kinematic encoding in LC units (`eph_01`–`eph_03`), spatial organization (`eph_04`),
  timing (`eph_05`), within-trial vs ITI bouts (`eph_07` planned; 103 units, go-responsive bout
  Δ +3.2 Hz vs ITI +0.1 Hz).
- **Recommendation:** keep this paper behavioral and cite the LC paper; reserve the ephys for a
  second paper unless a reviewer-proof neural hook is wanted. See decision 5.

### Supplementary figures
- S1 Noise criteria: duration floor (10 frames at 50 Hz cutoff: 9.7% of detections), single-frame
  2.3%, spatial outliers r_max_norm > 4×: 0.5%; velocity–distance 2-frame degeneracy; video
  ground-truth frames (`kin_00`).
- S2 Filter-threshold sensitivity curves (`kin_01` §3b).
- S3 Lickometer parameter-sweep heatmaps and ILI distributions (`tongue_lickometer`).
- S4 Multi-lick movements and licks-without-movements examples (`kin_05` §4–§5).
- S5 Per-session versions of Figs 4c, 4e; KS table; bootstrap SD CIs.
- S6 Per-mouse decode AUCs and value-screen matrices.

---

## 5. Discussion (points to make)

1. **RT in lick tasks is a composite.** The video first-movement latency and the lickometer RT
   differ by an integer number of ~170 ms sub-movement intervals. Studies that regress neural
   activity on lickometer RT (including the companion LC paper, which defines RT as first tongue
   detection with a 1 s cutoff) are mixing latency with sub-movement count. Propose reporting
   RT₁ and k separately.
2. **"Preparatory" activity may be movement.** Pre-lick tongue kinematics predict choice at
   high AUC; any preparatory-period neural signal in a lick task must be checked against overt
   tongue motion (cf. uninstructed-movement literature, Musall et al. 2019, Stringer et al. 2019;
   ALM preparatory activity).
3. **Non-contact movements are a behavioral class.** Shorter, slower, jaw-proximal; prevalent
   in ~50% of trials before the response. Candidate readouts: hesitation, deliberation,
   arousal; their count correlates with lick count per trial (r = 0.69), suggesting shared
   engagement drive.
4. **Vigor is multidimensional.** Latency, sub-movement count, peak velocity, and extent
   dissociate (Fig 4f: spread constant across k). This is the kinematic vocabulary the
   neuromodulator program needs (LC-NE → RT; DA → velocity, per the grant framing).
5. **Relation to the tongue-control literature.** Sub-movements here are preparatory and
   choice-bearing, not target-correction (Bollu 2021); 2D bottom view vs 3D; decision task vs
   drinking.
6. **Limitations.** 2D tracking; lickometer as imperfect ground truth; head-fixed; movement
   segmentation depends on confidence masking; noise criterion choice.

---

## 6. Methods (sections, with the numbers already fixed)

- Animals, task, sessions (15 mice, 44 sessions; training stage; ephys vs non-ephys sessions).
- Video: FLIR Blackfly S, 500 Hz, 720 × 540, bottom view; camera–behavior clock alignment
  (`video_alignment`).
- Pose estimation: Lightning Pose, 11 keypoints, ~1.4k labeled frames; test pixel error per
  keypoint; likelihood threshold 0.90.
- Kinematic processing: cubic interpolation of masked gaps, 4th-order 50 Hz Butterworth
  (zero-phase), velocity by differentiation; segmentation with ≤ 3 dropped frames; outbound
  phase = onset → max excursion.
- Movement-level features (the 49 columns): duration, peak/mean velocity, distance, endpoint,
  excursion angle, jaw-relative extrema, outbound metrics.
- Lick alignment: contact ↔ frame within 10 ms; cue-response movement; ordinal k;
  `movement_before_cue_response`; latency definitions (movement latency from go, lick latency).
- Session inclusion and movement-level noise criterion.
- Video lick detection and lickometer validation metrics (50 ms window, refractory 100 ms).
- RT decomposition: Δt estimate, de-shift, KS, bootstrap SD.
- Choice decoding: ridge-logistic, train/test split within session, permutation importance,
  session-level pooling.
- Behavioral model latents (MLE fits; Q, chosen probability, RPE) and screens.
- Statistics: session as sampling unit for population claims; effect sizes with CIs.
- Code/data availability: `aind-dynamic-foraging-behavior-video-analysis`, this capsule.

---

## 7. Literature gaps this work fills

| Existing work | What it established | Gap your data fills |
|---|---|---|
| Bollu et al. 2021 *Nature*; Xu et al. 2022 *Nature*; collicular map 2024 *Nature*; interception preprint 2026 | Licks are reach-like with corrective submovements; ALM/SC control tongue direction | Tongue kinematics inside a value-based *choice* task, over 44 sessions and 246k movements; sub-movements that are preparatory/choice-bearing, not corrective |
| Lickometer device papers (J Neurophysiol 2019; eNeuro 2024 optical; capacitive 2025) | Validate contact detectors against video | Quantify what contact detection *cannot* see: 39% of tongue movements, ~50% of trials with pre-response movement |
| RT literature in mice (motor-readiness 2023; "humanlike RT" preprint 2024; LATER/log-normal models) | RT distributions and readiness | RT = RT₁ + (k−1)·Δt; constant within-k spread; video vs lickometer RT are different variables |
| ALM preparatory activity (Li 2015/2016; Inagaki 2019); uninstructed movements (Musall 2019; Stringer 2019) | Neural choice signals before the lick; movements explain neural variance | Overt tongue kinematics predict choice before contact (AUC 0.95) — a behavioral confound/mechanism for "preparatory" signals |
| Vigor & value (saccade/reach vigor and RPE; Hughes 2020; Hamilos 2021; Nat Neurosci 2025) | Reward modulates vigor of limb/eye movements; DA links debated | Trial-level tongue vigor vs model-derived Q/RPE and post-outcome state in a foraging task; a multidimensional vigor definition |
| Companion LC paper (Su et al.) | LC-NE neurons encode RT and vigor; RT defined from first tongue detection | Behavioral foundation and refinement of the RT variable those neural analyses use |

---

## 8. Open decisions before drafting

1. **Movement-level noise criterion.** `kin_01`'s `out_duration > 0.05 s` removes 61% of
   detections; `kin_00`'s physically grounded criterion (duration floor + spatial outlier)
   removes 10%. The paper needs one; recommend `kin_00`'s and report sensitivity in S2.
2. **Pooling.** Figures 5 and 6 are single-session. Port to `kin_06`/`kin_07` and pool with
   session (and mouse) as the sampling unit before any claim goes in the abstract.
3. **RT definition alignment with the LC paper.** Report both `movement_latency_from_go` and
   `lick_latency`, state which the LC paper used, and show how k affects each.
4. **Cue-response definition** for trials where the first contact is ambiguous (multi-lick
   movements, movements spanning the go cue — currently nulled).
5. **Ephys in or out.** Behavioral paper + cite LC paper (recommended), or add Figure 8.
6. **Model-latent availability** per session (`get_mle_model_fitting`); sessions without fits
   drop from Figure 6 — count them now.
7. **Mice metadata**: sex, age, training history, and which sessions carried ephys.

## 9. Suggested order of remaining analyses (value ÷ effort)

1. `kin_04` §6 post-reward vigor (pooled parquet only) → Figure 6a.
2. `kin_06` pooled choice decode → Figure 5d population AUC.
3. `kin_03` UMAP conclusion → Figure 3d.
4. `kin_04` session timecourse quantified → Figure 7.
5. `kin_00` criterion decision + rerun `kin_01`/`kin_02`/`kin_05` under it.
6. `kin_07` value screen (Code Ocean, Han's pipeline) → Figure 6b–e.
7. Pooled licks-without-movements (`kin_05` §4) → Figure 2b completion.

## 10. Key references to collect

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
