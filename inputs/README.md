# inputs

Small tables that scripts in `code/` read as inputs, versioned with the code so every build
records exactly which list it used.

| File | Contents | Made by | Read by |
|---|---|---|---|
| `me_sessions_fip_curated.csv` | 301 FIP sessions across the two curated FIP assets: `asset` (`3ch` = `DANE_3channels_curated`, `4ch` = `DA_NE_4channels`), `subject`, `ses_idx`, `raw_session` (raw behavior asset name), `has_same_side_pair`, `used_in_fip05_07` (97) | Looping over the curated FIP data assets, before 2026-09-29; the script is not in this repo | `build_me_asset_map.py`, `build_me_table.py` |
| `me_sessions_fip_curated.json` | The same 301 `raw_session` names as a plain list | As above | — (session-list format the ME batch launcher takes) |
| `me_assets_fip.csv` | The 97 `used_in_fip05_07` sessions → their motion-energy result: `me_asset_id`, `me_asset_name` (S3 folder), `me_run_id`, `raw_asset_id` | `build_me_asset_map.py`, from the ME batch manifests (runs `ce719ee8`, `2b3a9315`, `342a45f7`) | `build_me_table.py`, `check_leading_lost_frames.py` |

See `code/fip_me_aligned_table_plan.md` for how these feed the aligned ME table.
