# Isosbestic correction: what it does to the curated traces

Does the pipeline's isosbestic motion correction remove movement artifact from DA and NE, or add noise, and
what does that do to the ME analyses (`fip_10` §2, `fip_12` §1, `fip_13`).
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
   Include the subtraction arithmetic as a worked example (notebook §2a): per band,
   var(G − β·iso) = var(G) − 2β·cov(G, iso) + β²·var(iso); the middle term is removal and needs coherence,
   the last is added whatever iso holds. Ask upstream why PR #38 (2025-03) switched from subtracting the
   0.05 Hz-filtered iso to the unfiltered one; subtracting the filtered iso would keep the slow removal
   without the fast addition.
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

## 4. Results (`fip_18_iso_correction`, first run 2026-11-16, 85 sessions / 9 animals)

Local run on the 3ch/4ch assets with the public NWBs of `DANE_bilateral` (aind-fip-dff 19.0); 3 of the 88
ME sessions have no public NWB (818585 2026-01-20, 818586 2026-01-02 and 2026-01-30). Plan items 1–5 done
except the `fip_13` regressions; what each found:

1. **The pipeline is understood.** The corrected trace is reproduced exactly from the NWB's bleach-only G
   and Iso and the QC's β (max difference 7e-17). The QC JSON carries everything per fiber and channel
   (F0 terms, calibration ratio, β, noise inflation); Rachel's curation columns (`snr`, `skewness`,
   `kurtosis`, `drop_reason`) use none of it. The evaluation status is **Fail for all 85 sessions** (worst
   of the four fibers wins), so it cannot serve as a filter.
   - **DA fiber: β = 0 in 41 of 85 fiber-sessions** (818586 20/20, 808054 7/15, 809487 6/8); median β 0.03.
     Half the curated DA is uncorrected and nothing in `df_fip` says which.
   - **NE fiber: β median 1.11 (IQR 0.83–1.49), β = 0 in 3**; noise inflation median 1.50, up to 2.5, rising
     linearly with β.
   - The curated `data` of the local 3ch/4ch assets (built from the 2025-12 processing, aind-fip-dff v14)
     differs from the public NWB's corrected trace (v19) in 32 of 85 sessions (r < 0.99); four of 808054's
     (2025-09-03 to 09-08) barely resemble it (r 0.02–0.56). The two processing versions do not give the
     same traces; on Code Ocean with `DANE_bilateral` the curated trace is the v19 one.
2. **Iso physics on our sessions matches the video-latents result.** On the NE fiber Iso's spectrum is flat
   above 0.2 Hz at the level of R and of G's own noise floor (at 1–3 Hz Iso has 0.9× G's power); coherence
   with G (median over animals) is 0.14 at 0.01–0.05 Hz, 0.09 at 0.05–0.2, 0.06 at 0.2–1 and at the
   estimator's floor (0.027 vs 0.023) at 1–3 Hz; β fitted within each band above 0.05 Hz is ≤ 0; the
   corrected/bleach-only power ratio is 0.98, 1.07, 1.32 and **2.29 at 1–3 Hz**. On the DA fiber the ratio
   is 1.0 everywhere (β ≈ 0), though Iso × G coherence is 0.12–0.22 below 1 Hz.
   - Worked example (818585 2025-12-16, β 1.19): at 1–3 Hz removed 0.007 %², added 0.285 %², net +0.277 on
     a bleach-only variance of 0.199 (×2.4); at 0.05–0.2 Hz (coherence 0.39) net −0.38. Subtracting the
     0.05 Hz-filtered iso keeps that removal and adds nothing above 0.2 Hz; a phase-shuffled iso adds the
     same as the real one in every band — the addition is set by var(iso) alone.
3. **Movement is barely in the PL iso.** NE fiber: the movement-locked part of Iso (its average at ME
   onsets) is a −0.05 % dip (p = 0.004); R +0.04 % at the onset, ~+0.1 % over 1–3 s; G +1.1 %; Iso × ME
   peak r −0.02 (ns). Coherence at the floor and a −0.05 % average dip are the same fact: the movement-locked
   part is ~1 % of Iso's power (0.05 % against a ~0.5 % trial-by-trial SD at 1–3 Hz). So β·Iso subtracted
   at an onset is ~99 % Iso's own fluctuation; the correction changes NE's movement response by +0.05 %
   (1.10 → 1.16 %, the dip removed — whether NE had it is unknowable, and R rises where Iso dips) while
   adding that noise (fig. 3e–f). At the go cue and outcome
   the PL controls move more (R 0.2–0.4 %, Iso 0.05–0.1 %; fig. 4b), still 10× below NE. This tests only
   artifact visible at 415 nm (and 560 nm); a 470 nm-specific artifact needs a sensor-dead recording.
   NAc fiber: Iso × ME peak r 0.20 (p = 0.012) and +0.15–0.3 % at ME onsets with wide spread across
   animals — the NAc iso does follow movement, but β ≈ 0 means the pipeline removes none of it.
4. **Fiber events.** One fiber-session above 30 robust SD: 809491 2025-10-20, DA fiber (126 SD, 65 s above
   10 SD, at 49.9 min; every channel), the session the video-latents work found; the ± 30 s rule removes
   2.5 % of it. Peak |iso| median 5.3 SD, p99 16.
   - **A step ~3 s after every task event** (go cue, rewarded, unrewarded) is in **every channel of both
     fibers** — Iso, R and G alike (fig. 4b), and Iso/R also spike at the event itself. Common to all
     channels means optical or acquisition, not the sensors. Source unknown; a rig-side question.
5. **Downstream.** Bleach-only vs corrected, per animal: NE × ME peak r 0.64 → 0.70 (8/9 animals up,
   p = 0.008), its lag 0.18 → 0.10 s; DA × NE peak r 0.27 → 0.31 (9/9, p = 0.004); DA measures unchanged
   (β ≈ 0); NE at ME onsets +0.03 z (ns). The correction lowers every coupling measure that involves NE,
   as added noise would.

Decisions still open: (a) switch the NE target to bleach-only (the numbers above argue for it; `fip_10`,
`fip_12`, `fip_13` would rerun), (b) the fiber-event rule (one session affected), (c) what to file on
`aind-fip-dff`: β = 0 unlabelled, iso noise added above 0.2 Hz (subtracting the 0.05 Hz-filtered iso would
avoid it), the 3 s common-mode step.

## 5. Notebook and documentation

One notebook, `fip_18_iso_correction`, Code Ocean plus local (`.venv-fip`; the NWBs read from public S3,
cached under `scratch/fip_18/`). Each section is one plan item; figures listed are the ones to keep.

| § | Content | Figure |
|---|---|---|
| 0 | Methods note: §1 of this file, condensed, with the commit link; the stage names in the NWB | — |
| 1 | Pipeline QC and parameters, 97 sessions | β per fiber-session (strip by mouse, DA vs NE fibers, β = 0 marked); F0 terms kept; QC status counts |
| 2 | Iso physics | Power spectra iso / G / R per fiber type with noise floors; coherence by band; β by band vs the pipeline's; power ratio corrected / bleach-only by band |
| 2a | Why subtracting adds: one PL session | (i) 20 s of bleach-only NE, β·iso and corrected NE, raw and band-passed 1–3 Hz, with the band's variance of each in the legend; (ii) the three terms of the variance identity per band (removed = 2β·cov, added = β²·var(iso), net) as grouped bars across 0.01–0.05, 0.05–0.2, 0.2–1, 1–3 Hz, next to coherence per band; (iii) the same identity with iso low-passed at 0.05 Hz before subtraction (what the pipeline did before PR #38) and with a synthetic iso of matched power but independent noise, to show the addition is set by var(iso) alone. Same for one NAc session, where β ≈ 0 makes every term vanish |
| 3 | Iso and red vs ME | xcorr and ME-onset averages for iso, red, corrected and bleach-only sensor, per mouse; residual-artifact panel (corrected − bleach-only at ME onsets against iso) |
| 4 | Fiber events and the post-event step | Peak robust-SD per iso trace (distribution, threshold); example sessions; event-triggered iso/DA/NE at go cue and outcome out to 6 s for the step |
| 5 | Downstream | Key numbers of `fip_10` §2, `fip_12` §1, `fip_13` for corrected vs bleach-only, per animal, same figures as those notebooks |

Documentation: CLAUDE.md row for `fip_18`; CHANGELOG; this file's §3 updated with results and decisions;
anything that turns out to be a pipeline defect (β = 0 unlabelled, iso events unflagged, the noise added at
1–3 Hz) goes to `aind-fip-dff` as an issue with the figure.
