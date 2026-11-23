# FIP sessions with usable motion energy

Made by `code/me_session_accounting.py` from `inputs/me_sessions_fip_curated.csv` and
`inputs/video_screen_fip.csv`; rerun it after either changes.

301 FIP sessions from 15 mice (the `DANE_3channels_curated`
and `DANE_4channels_curated` assets). Every session has an ME run, so a camera's ME is usable when
its video passes the timing and image-quality screen: 264 bottom cameras and 236 side
cameras.

## Usable ME, sessions per mouse

| Subject | Sessions | Bottom ME usable | Side ME usable | Both cameras usable | DA + LC (same side) | DA + LC, bottom ME | Analysis set | Analysis set, bottom ME |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 800886 | 15 | 14 | 14 | 14 | 5 | 4 | 5 | 4 |
| 808054 | 19 | 18 | 18 | 18 | 17 | 16 | 16 | 15 |
| 808057 | 20 | 19 | 18 | 18 | 0 | 0 | 0 | 0 |
| 809487 | 14 | 13 | 0 | 0 | 8 | 8 | 8 | 8 |
| 809491 | 22 | 21 | 21 | 21 | 20 | 19 | 20 | 19 |
| 813929 | 16 | 13 | 13 | 13 | 0 | 0 | 0 | 0 |
| 815334 | 23 | 19 | 19 | 19 | 6 | 5 | 4 | 3 |
| 816212 | 19 | 11 | 16 | 11 | 8 | 5 | 8 | 5 |
| 816214 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 816883 | 20 | 19 | 17 | 17 | 0 | 0 | 0 | 0 |
| 818580 | 19 | 15 | 16 | 15 | 0 | 0 | 0 | 0 |
| 818585 | 32 | 30 | 30 | 30 | 8 | 7 | 8 | 7 |
| 818586 | 23 | 22 | 4 | 4 | 23 | 22 | 23 | 22 |
| 820688 | 31 | 25 | 25 | 24 | 1 | 1 | 0 | 0 |
| 820691 | 23 | 20 | 20 | 20 | 1 | 0 | 0 | 0 |
| **ALL** | **301** | **264** | **236** | **229** | **102** | **92** | **97** | **88** |

- **DA + LC (same side)**: a curated lateral-NAc DA fiber (`latNAcc(X)-DA`) and a curated PL
  LC-axon fiber (`PL(X)-LCAxonCa`) on the same hemisphere (`fu.inventory_pairs`).
- **Analysis set**: the sessions `fip_10`–`fip_17` use: one hemisphere per mouse (the side with
  more sessions) and at least 3 sessions on it (`fu.choose_side`). With bottom-camera ME, these
  are the movement sessions of the `fip_*` notebooks (`fu.add_motion_energy`).

## Excluded cameras, by QC category

| Subject | Bottom: clock step | Bottom: trigger count | Bottom: clock rate | Bottom: blurred | Bottom: image changed | Side: clock step | Side: trigger count | Side: unreadable | Side: overexposed | Side: blurred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 800886 | 1 |  |  |  |  | 1 |  |  |  |  |
| 808054 | 1 |  |  |  |  | 1 |  |  |  |  |
| 808057 | 1 |  |  |  |  | 1 |  | 1 |  |  |
| 809487 | 1 |  |  |  |  | 1 |  |  | 13 |  |
| 809491 | 1 |  |  |  |  | 1 |  |  |  |  |
| 813929 | 3 |  |  |  |  | 3 |  |  |  |  |
| 815334 | 4 |  |  |  |  | 4 |  |  |  |  |
| 816212 | 3 | 3 | 1 |  | 1 | 3 |  |  |  |  |
| 816214 |  |  |  |  |  |  |  |  |  |  |
| 816883 | 1 |  |  |  |  | 1 |  |  | 2 |  |
| 818580 | 3 | 1 |  |  |  | 3 |  |  |  |  |
| 818585 | 2 |  |  |  |  | 2 |  |  |  |  |
| 818586 |  | 1 |  |  |  |  | 1 |  | 18 |  |
| 820688 | 4 | 1 |  | 1 |  | 4 | 1 |  |  | 1 |
| 820691 | 3 |  |  |  |  | 3 |  |  |  |  |
| **ALL** | **28** | **6** | **1** | **1** | **1** | **28** | **2** | **1** | **33** | **1** |

Timing QC (from the video-analysis library's timing check):

- **clock step**: the Harp frame times jump (`harp_evenly_spaced`). The cameras share the Harp
  clock, so a clock step fails both cameras of a session.
- **trigger count**: the camera's trigger log and its frame numbers disagree (`trigger_log_count`).
- **clock rate**: the camera and Harp clocks run at different rates (`clock_rates_agree`).
- **unreadable**: the video could not be read.

Image-quality QC (median over reference frames):

- **overexposed**: too many pixels clipped at the top of the range (`pct_clipped_high` > 3.75 %).
- **blurred**: sharpness far below the camera's usual (`sharpness_dev` ≤ 0.45).
- **image changed**: the view changes during the session (`similarity_p5` < 0.998).
