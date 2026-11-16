# Isosbestic controls to-do

Is any of DA's or NE's relation to motion energy (`fip_10` §2, `fip_12` §1, `fip_13`) motion artifact left
after motion correction? Asked for by Kyle Severson. Split out of `fip_todo.md` on 2026-11-16; the rest of the
FIP and ME work stays there.

---

## How the curated traces are corrected

The traces in the curated `df_fip.parquet` (`data` column, `dff-bright_mc-iso-IRLS`) come from the
`aind-fip-dff` capsule, run inside the "Combined Dynamic Foraging + fiber pipeline" on Code Ocean. Rachel's
capsule picks that variant and curates; it does no signal processing. For `DANE_bilateral` (asset 4686d370)
all 295 sessions were processed on 2026-11-10/11 with release 19.0, GitHub commit
[`c1733bb`](https://github.com/AllenNeuralDynamics/aind-fip-dff/tree/c1733bb).

Per fiber, each channel (G, Iso, R) separately:

1. **Baseline** (`tc_brightfit`): F0 = b_inf · (1 + up to three decaying exponentials) · (1 − a rising
   brightening exponential), fitted to the whole session with robust reweighting that follows the lower edge
   of the trace. ΔF/F = F / F0 − 1. The first 5 s are cut before the fit and filled with a constant.

Then across the fiber's channels (`motion_correct`):

2. All ΔF/F traces low-passed at 0.05 Hz, for the fit only.
3. Each of G and R regressed on Iso (robust, `AsymmetricTukeyBiweight(2)`); slope β clipped at ≥ 0;
   intercept discarded.
4. β · (unfiltered Iso ΔF/F, mean removed) subtracted from the channel; Iso set to 0.
5. Result low-passed at 3 Hz (second-order Butterworth, zero-phase).

β is the gain from Iso to the channel, learned only from what they share below 0.05 Hz (periods over 20 s)
and applied at every frequency.

## Data

The curated `df_fip` holds only the corrected sensor channels (e.g. 816214 2025-12-03: `G_1` NE, `G_3` DA),
with no iso or raw traces. The processed NWBs hold every stage for every channel: `<ch>_<fiber>_dff-bright`
(baseline-corrected) and `<ch>_<fiber>_dff-bright_mc-iso-IRLS` (corrected), plus the raw traces and the
`dff-qc/` figures (fit, and Iso-vs-channel scatter with the fitted line). Each session's NWB path is in
`df_sess_<mouse>.csv`, column `s3_location` (public `s3://aind-open-data/`).

To compare iso with the corrected channels, put iso through the same 3 Hz low-pass.

## Open

Do not use motion-corrected iso: the correction sets it to 0. Residual artifact is the part of a channel's
artifact that iso does not contain. Instead:

1. Raw iso (`Iso_<fiber>_dff-bright`, no `mc`) × ME through the same xcorr and ME-onset averages: how large
   the movement artifact is per fiber.
2. Corrected DA/NE and raw iso averaged at ME onsets: an iso-shaped deflection (same timing and sign) left
   in corrected NE marks residual artifact.
3. If the NE fiber's red channel carries no sensor, red × ME measures artifact. Use `R_<fiber>_dff-bright`:
   the `mc` version has had iso regressed out.
4. Activity-independent recordings (GFP or a mutant sensor) through the same pipeline, if any exist.
5. β by frequency band. Fit G (and R) on Iso as `motion_correct` does, but band-passed at 0.5–3 Hz, and in
   windows around ME onsets, from the `dff-bright` traces. Compare with the pipeline's slow (< 0.05 Hz) β per
   fiber and session. If they differ, the correction is miscalibrated at the timescale of movement:
   under-corrected if the fast β is larger, over-corrected if smaller. The slow β may be set by leftover
   bleaching or blood volume, the fast one by licking.
