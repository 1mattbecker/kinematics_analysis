# Isosbestic correction: what it does to the curated traces

Does the pipeline's isosbestic motion correction remove movement artifact from DA and NE, or add noise, and
what does that do to the ME analyses (`fip_10` §2, `fip_12` §1, `fip_13`)? Asked for by Kyle Severson.
Technical questions only; the biology stays in the video-latents repo. Split out of `fip_todo.md` 2026-11-16.

---

## 1. How the curated traces are made

`df_fip.data` is `dff-bright_mc-iso-IRLS` from `aind-fip-dff` release 19.0, GitHub commit
[`c1733bb`](https://github.com/AllenNeuralDynamics/aind-fip-dff/tree/c1733bb), run in the "Combined Dynamic
Foraging + fiber pipeline" on 2026-11-10/11 for all 295 `DANE_bilateral` sessions (asset 4686d370). Rachel's
capsule picks that variant and curates; it does no signal processing.

Per channel (G, Iso, R) of each fiber — `tc_brightfit`:
- F0 = b_inf · (1 + up to three decaying exponentials) · (1 − a rising "brightening" term), one fit per
  session, τ ≥ 1 s so it cannot follow events. Robust refit follows the lower edge of the trace: the residual
  scale is the MAD of the *negative* residuals, positive residuals are inflated by their skewness, Tukey c = 3.
- ΔF/F = F / F0 − 1. The first 5 s are cut before the fit and filled with a constant.

Per fiber — `motion_correct`:
1. All ΔF/F traces low-passed at 0.05 Hz, for the fit only.
2. G and R each regressed on Iso (robust, `AsymmetricTukeyBiweight(2)`: c = 2 above the line, 4.685 below,
   so the fit follows the lower edge of the channel-vs-iso cloud; reason in issue #50, value 2 unexplained).
3. Slope β clipped at ≥ 0 (a negative β would add iso to the channel); intercept discarded. **β = 0 means no
   correction at all, still labelled `mc-iso-IRLS`.**
4. β · (unfiltered iso ΔF/F, mean removed) subtracted; iso set to 0.
5. Result low-passed at 3 Hz (2nd-order Butterworth, zero-phase; added after issue #50 because a weak iso
   made the corrected trace noisy).

So β is learned from what iso and the channel share below 0.05 Hz (periods > 20 s) and applied at every
frequency. The pipeline's own QC (`dff-qc/quality_control.json`, README): F0 calibration ratio, baseline level
at start/middle/end, noise inflation by the correction, pre-GoCue drift, auto-fail at β > 10. Nothing on iso
events or on β = 0.

**Data.** The processed NWBs (path in `df_sess_<mouse>.csv`, `s3_location`, public `aind-open-data`) hold
every stage for every channel: raw, `<ch>_<fiber>_dff-bright` (bleach-only) and `_dff-bright_mc-iso-IRLS`,
plus `dff-qc/` figures and the run log with every fitted parameter. Bleach-only traces must get the same 3 Hz
low-pass before comparison with the curated ones.

## 2. What the video-latents work found (branch `nm-bench`, development mice, 2026-11)

Technical findings only. Fibers there: NAc = DA (dLight, green) + ACh (red); PL = LC-axon GCaMP (NE), red
sensor-free. 133 PL and 342 NAc fiber-sessions.

| Finding | Where | Concern |
|---|---|---|
| **PL iso is measurement noise.** Flat spectrum 0.2–9 Hz at the level of red and of NE's own noise floor above 3 Hz; coherence with bleach-only NE 0.038 at 0.03–0.2 Hz, at the floor above. | `iso_physics.py` | The correction cannot be removing artifact at event timescales |
| **β comes from slow drift.** 0.7–1.9 per mouse at PL, equal to the slope of NE on iso below 0.01 Hz. | same | Confirms §1: a gain learned on drift, applied to noise |
| **The subtraction adds noise to NE.** Power ×2.9 at 1–3 Hz, ×1.4 at 0.2–1 Hz. Corrected NE is 27 % iso overall, 50 % at 1–3 Hz (8/8 mice); bleach-only NE depends on iso by 0.02. | same; `w3_iso` | **Main processing issue.** Curated NE carries iso noise at the frequencies of the ME analyses |
| **At NAc the correction does nothing.** β ≈ 0 in 10/11 mice, so corrected DA = bleach-only DA. DA has 100–400× iso's power below 3 Hz; coherence 0.20 (0.03–0.2 Hz), 0.12 (0.2–1 Hz). | same | Not a problem in itself; but the two sensors are treated differently without any flag in `df_fip` |
| **Bleach-only NE is better explained by ME** (0.29 vs 0.25 over task); video gains survive iso as a covariate. | `w3_iso` | Argues for the bleach-only trace |
| **Iso carries no sensor signal** at go cue / outcome (DA 19–24 % vs iso 0.1 %, wrong shape; NE 1.5 % vs −0.04 %). | `iso_events.py` | Rules out the "iso is not isosbestic" worry for DA and NE |
| **Iso does follow task and movement within a session** (task model up to 0.17 of NAc iso; video 0.17 within session, ~0 across mice; slow 30 s movement filters raise PL iso by 0.07–0.09). PL red too (~0.14). Artifact grows below 0.03 Hz; below 0.003 Hz bleaching residue dominates. | §4 | Movement artifact exists, is slow, and differs by mouse — so the slow β is not nonsense, only misapplied |
| **How much artifact iso could hide in NE:** ≤ 5–7 % of power per band, if the artifact is no larger at 470 than 415 nm. | `iso_physics.py` | A 470-only artifact is not excluded by any channel we have |
| **Fiber events in iso:** 6 traces in 4 sessions with 58–105 robust-SD events (all-trace median peak 6.4, p99 ≈ 15); e.g. 809491 2025-10-20 ~1 min on every NAc channel; 815334 2025-10-30 latNAcc L > 10 SD for 115 s. Not flagged by the pipeline QC. | `fiber_events.md` | **Curation gap.** Proposed rule (not applied): drop the fiber-session above ~30 SD, or mask ± 30 s |
| **A ~0.2 % step ~3 s after every event** in ACh and iso alike. Source unknown. | `iso_events.py` | Rig or acquisition effect; could sit in DA/NE too |
| A shift null does not test movement coupling with the ridge model (every signal gets p = 1/51). | §5 | Methods, not processing; noted so it isn't reused here |

Recommendation there: use bleach-only NE (`NE.nomc`). **Not yet decided for this repo.** Whether the
development mice overlap the 9 FIP×ME animals here is not checked.

## 3. Plan

Order: establish the facts on *our* 97 sessions, then decide the trace, then rerun. Decisions to take at the
end, not before: (a) corrected vs bleach-only per sensor; (b) a fiber-event rule for curation; (c) what to
tell Rachel / file on `aind-fip-dff`.

1. **Pipeline QC and fitted parameters for the curated sessions.** Pull `quality_control.json`, the run
   logs' fitted parameters (F0 terms kept, β per channel) and the `dff-qc` figures for the 97 FIP×ME
   sessions. Count β = 0 fiber-sessions (never corrected) and β > 10 fails; check which QC statuses Rachel's
   curation uses. Avoids redoing what the pipeline measures.
2. **Replicate the iso physics on our sessions.** Iso spectrum against each channel's noise floor; coherence
   iso × bleach-only channel by band; β by band (slow < 0.05 Hz as the pipeline, 0.5–3 Hz, and in windows
   around ME onsets); power added by the correction per band. Per fiber and session; DA and NE fibers.
3. **Iso and red against ME.** Iso (and PL red) × ME through the same xcorr and ME-onset averages as
   `fip_10` §2 / `fip_12` §1: how large the movement artifact is per fiber, at what timescale, and how
   mouse-specific. Then corrected and bleach-only NE/DA at ME onsets next to iso: an iso-shaped deflection
   left in the corrected trace marks residual artifact; iso-shaped *noise* added to it marks the problem in §2.
4. **Fiber events.** Detect large iso events (robust SD) in all curated sessions; cross with Rachel's
   curation columns and with the ME table's own exclusions; tally sessions affected; apply the proposed rule
   and see what it removes. Also the ~3 s post-event step: is it in our iso, DA, NE; rig- or date-specific.
5. **Downstream impact.** Rerun `fip_10` §1–2, `fip_12` §1 and `fip_13` on bleach-only traces (same 3 Hz
   low-pass) next to the curated traces, with and without the fiber-event rule: which numbers move. This is
   the input to decision (a).
6. Open, no plan yet: a sensor-dead control at PL (GFP / mutant sensor) through the same pipeline, if any
   exist; ACh has no same-wavelength reference on the NAc fiber.

## 4. Notebook and documentation

One notebook, `fip_18_iso_correction`, Code Ocean plus local (`.venv-fip`; the NWBs read from public S3,
cached under `scratch/fip_18/`). Each section is one plan item; figures listed are the ones to keep.

| § | Content | Figure |
|---|---|---|
| 0 | Methods note: §1 of this file, condensed, with the commit link; the stage names in the NWB | — |
| 1 | Pipeline QC and parameters, 97 sessions | β per fiber-session (strip by mouse, DA vs NE fibers, β = 0 marked); F0 terms kept; QC status counts |
| 2 | Iso physics | Power spectra iso / G / R per fiber type with noise floors; coherence by band; β by band vs the pipeline's; power ratio corrected / bleach-only by band |
| 3 | Iso and red vs ME | xcorr and ME-onset averages for iso, red, corrected and bleach-only sensor, per mouse; residual-artifact panel (corrected − bleach-only at ME onsets against iso) |
| 4 | Fiber events and the post-event step | Peak robust-SD per iso trace (distribution, threshold); example sessions; event-triggered iso/DA/NE at go cue and outcome out to 6 s for the step |
| 5 | Downstream | Key numbers of `fip_10` §2, `fip_12` §1, `fip_13` for corrected vs bleach-only, per animal, same figures as those notebooks |

Documentation: CLAUDE.md row for `fip_18`; CHANGELOG; this file's §3 updated with results and decisions;
anything that turns out to be a pipeline defect (β = 0 unlabelled, iso events unflagged, the noise added at
1–3 Hz) goes to `aind-fip-dff` as an issue with the figure.
