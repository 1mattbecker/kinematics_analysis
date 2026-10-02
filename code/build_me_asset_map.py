# /// script
# requires-python = ">=3.10"
# dependencies = ["codeocean>=0.16,<0.17", "pandas", "requests"]
# ///
"""Map each FIP session to its motion-energy result asset.

Writes ``inputs/me_assets_fip.csv``: one row per ``used_in_fip05_07`` session
in ``inputs/me_sessions_fip_curated.csv``, with ``raw_session``,
``me_asset_id``, ``me_asset_name``, ``me_run_id`` and ``raw_asset_id`` (the
behavior asset the ME run read).

The mapping comes only from the batch launcher's manifests
(``aind-motion-energy-batch-capsule/results/batch_manifest_<run id>.json``),
never from "newest result by tag": ~31 leftover test results share the tag.
Runs are applied in ``RUNS`` order, so a later run (the ``.mp4`` rerun of the
10 808054 2025-09-02 to 09-15 sessions) replaces an earlier one.

Asset names come from the Code Ocean API, through the launcher's own
``scripts/trigger_batch.make_client`` (token from ``API_SECRET``); the name is also
the result folder under ``s3://aind-scratch-data/matt.becker/motion_energy/``.
Each camera's ``*_me_metadata.json`` is read anonymously over HTTPS and must
have a ``.mp4`` ``video_path`` and a null ``end_frame``.

Usage::

    uv run code/build_me_asset_map.py [--manifests <dir>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SESSIONS_CSV = REPO / "inputs" / "me_sessions_fip_curated.csv"
OUT_CSV = REPO / "inputs" / "me_assets_fip.csv"
BATCH_CAPSULE = REPO.parent / "aind-motion-energy-batch-capsule"
DEFAULT_MANIFESTS = BATCH_CAPSULE / "results"

# Applied in order; later runs win. ee91b1d6 was a dry run and is left out.
RUNS = [
    "ce719ee8-ca14-4f52-8f8b-8655237a3c8e",
    "2b3a9315-22b5-47f9-b929-687240392f82",
    "342a45f7-f157-47d6-b4dd-763438fbeadf",  # .mp4 rerun, 808054 09-02..09-15
]

SCRATCH_URL = "https://aind-scratch-data.s3.amazonaws.com"
ME_PREFIX = "matt.becker/motion_energy"


def list_keys(prefix):
    """Return the object keys under ``prefix`` in the scratch bucket."""
    keys, token = [], None
    while True:
        url = f"{SCRATCH_URL}/?list-type=2&prefix={prefix}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        text = urllib.request.urlopen(url).read().decode()
        keys += re.findall(r"<Key>(.*?)</Key>", text)
        token_match = re.search(
            r"<NextContinuationToken>(.*?)</NextContinuationToken>", text
        )
        if not token_match:
            return keys
        token = token_match.group(1)


def read_json(key):
    """Read one JSON object from the scratch bucket."""
    return json.load(urllib.request.urlopen(f"{SCRATCH_URL}/{key}"))


def main():
    """Build the session -> ME asset table and check each result's metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    args = parser.parse_args()

    sessions = pd.read_csv(SESSIONS_CSV)
    used = sessions.loc[sessions["used_in_fip05_07"], "raw_session"].tolist()

    # Session -> (run, record); later runs replace earlier ones
    chosen = {}
    for run in RUNS:
        manifest = json.loads(
            (args.manifests / f"batch_manifest_{run}.json").read_text()
        )
        for record in manifest:
            if record["session"] not in used:
                print(f"  not a FIP session, ignored: {record['session']} ({run[:8]})")
                continue
            if record["status"] != "completed":
                print(f"  ✗ {record['session']}: {record['status']} in {run[:8]}")
                continue
            if record["session"] in chosen:
                print(
                    f"  {record['session']}: {chosen[record['session']][0][:8]}"
                    f" replaced by {run[:8]}"
                )
            chosen[record["session"]] = (run, record)

    missing = [s for s in used if s not in chosen]
    if missing:
        sys.exit(f"No completed ME result for {len(missing)} sessions: {missing}")

    # Same client (domain, token) as the launcher that made the manifests
    sys.path.insert(0, str(BATCH_CAPSULE / "scripts"))
    from trigger_batch import make_client

    client = make_client()
    rows, problems = [], []
    for session in used:
        run, record = chosen[session]
        asset = client.data_assets.get_data_asset(record["result_asset_id"])
        if not asset.name.startswith(f"{session}_motionenergy_"):
            problems.append(f"{session}: asset name {asset.name}")
        keys = list_keys(f"{ME_PREFIX}/{asset.name}/")
        metadata_keys = [k for k in keys if k.endswith("_me_metadata.json")]
        if len(metadata_keys) != 2:
            problems.append(f"{session}: {len(metadata_keys)} me_metadata.json files")
        for key in metadata_keys:
            meta = read_json(key)
            camera = key.rsplit("/", 1)[1].removesuffix("_me_metadata.json")
            if not meta["video_path"].endswith(".mp4"):
                problems.append(f"{session} {camera}: video {meta['video_path']}")
            if meta["end_frame"] is not None or meta["start_frame"] is not None:
                problems.append(
                    f"{session} {camera}: frames {meta['start_frame']}"
                    f"..{meta['end_frame']}"
                )
            if meta["n_me_frames"] != meta["n_frames_decoded"] - 1:
                problems.append(
                    f"{session} {camera}: {meta['n_me_frames']} ME values for "
                    f"{meta['n_frames_decoded']} frames"
                )
        rows.append(
            {
                "raw_session": session,
                "me_asset_id": asset.id,
                "me_asset_name": asset.name,
                "me_run_id": run,
                "raw_asset_id": record["input_asset_id"],
            }
        )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(pd.DataFrame(rows)["me_run_id"].str[:8].value_counts().to_string())
    if problems:
        print(f"{len(problems)} problems:")
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("All ME metadata: .mp4 video, full length, N-1 values per camera")


if __name__ == "__main__":
    main()
