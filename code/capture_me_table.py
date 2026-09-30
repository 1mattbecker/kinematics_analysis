"""
capture_me_table.py — capture the built ME table as one Code Ocean data asset.

Run in the cloud workstation after ``build_me_table.py``. Checks the build is
complete (every session indexed, every ``ok`` camera's Parquet file present,
``build_info.json`` written), then creates a result data asset from the
workstation's output folder, the way the ME batch launcher
(``aind-motion-energy-batch-capsule/code/run_capsule.py::start_capture``)
captures a computation. The asset is stored externally in AIND scratch,
``s3://aind-scratch-data/matt.becker/fip_motion_energy_aligned/<name>/``, like
the ME results; ``--internal`` keeps it in Code Ocean instead.

Needs ``API_SECRET`` (a Code Ocean API token) and ``CO_DOMAIN`` (the Code Ocean
URL the ME batch launcher uses) in the environment, and
``codeocean<0.17``: the server is 4.7.3 and 0.17 needs >= 4.8, while the
capsule image has 0.17. Install the older SDK beside the image's, not into it::

    python -m venv /tmp/co016 && /tmp/co016/bin/pip install -q "codeocean>=0.16,<0.17"
    /tmp/co016/bin/python code/capture_me_table.py            # from /root/capsule

The asset is tagged ``fip-me-aligned``, not ``motion-energy``, so it never
mixes with the per-session ME results that share that tag.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from codeocean import CodeOcean
from codeocean.data_asset import (
    AWSS3Target,
    CloudWorkstationSource,
    DataAssetParams,
    DataAssetState,
    Source,
    Target,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "results" / "fip_motion_energy_aligned"
SESSIONS_CSV = REPO / "metadata" / "me_sessions_fip_curated.csv"
CAMERAS = ["BottomCamera", "SideCameraRight"]

SCRATCH_BUCKET = "aind-scratch-data"
SCRATCH_PREFIX = "matt.becker/fip_motion_energy_aligned"
TAGS = ["fip-me-aligned", "fip", "derived"]


def check_complete(out_dir):
    """Raise unless the build in ``out_dir`` covers every session; return counts."""
    with open(SESSIONS_CSV) as fh:
        sessions = {r["raw_session"] for r in csv.DictReader(fh) if r["used_in_fip05_07"] == "True"}
    with open(out_dir / "index.csv") as fh:
        index = list(csv.DictReader(fh))
    indexed = {(r["session"], r["camera"]) for r in index}
    missing = {(s, c) for s in sessions for c in CAMERAS} - indexed
    if missing:
        raise SystemExit(f"{len(missing)} session x camera rows missing from index.csv, e.g. {sorted(missing)[:3]}")
    ok = [r for r in index if r["status"] == "ok"]
    no_file = [r for r in ok if not (out_dir / r["session"] / f"{r['camera']}.parquet").exists()]
    if no_file:
        raise SystemExit(f"{len(no_file)} ok cameras have no Parquet file, e.g. {no_file[0]['session']}")
    if not (out_dir / "build_info.json").exists():
        raise SystemExit("build_info.json missing: the build did not finish")
    return len(sessions), len(ok), len(index) - len(ok)


def main():
    """Check the build, create the data asset and wait until it is ready."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Folder build_me_table.py wrote")
    parser.add_argument(
        "--workstation-id",
        default=os.environ.get("CO_COMPUTATION_ID"),
        help="Computation ID of this cloud workstation (default: $CO_COMPUTATION_ID)",
    )
    parser.add_argument("--domain", default=os.environ.get("CO_DOMAIN"), help="Code Ocean URL (default: $CO_DOMAIN)")
    parser.add_argument("--internal", action="store_true", help="Keep the asset in Code Ocean, not AIND scratch")
    parser.add_argument("--dry-run", action="store_true", help="Check the build and print the request only")
    args = parser.parse_args()

    n_sessions, n_ok, n_refused = check_complete(args.out)
    build_info = json.loads((args.out / "build_info.json").read_text())
    print(f"Build complete: {n_sessions} sessions, {n_ok} cameras ok, {n_refused} refused")
    print(f"  library {build_info['library_version']} @ {build_info['library_commit']}, repo {build_info['repo_commit']}")
    if not build_info["library_commit"] or str(build_info["repo_commit"]).endswith("-dirty"):
        raise SystemExit("build_info.json lacks the library commit or the repo had uncommitted changes")
    if not args.workstation_id:
        raise SystemExit("No workstation computation ID: pass --workstation-id")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    name = f"fip_motion_energy_aligned_{stamp}"
    params = DataAssetParams(
        name=name,
        tags=TAGS,
        mount="fip_motion_energy_aligned",
        description=(
            f"Per-frame corrected Harp time + me_clean for {n_sessions} FIP sessions "
            f"({n_ok} cameras ok, {n_refused} refused; see index.csv). "
            f"kinematics_analysis {build_info['repo_commit'][:7]}, "
            f"video-analysis {build_info['library_version']} ({build_info['library_commit'][:7]})."
        ),
        source=Source(
            cloud_workstation=CloudWorkstationSource(
                id=args.workstation_id,
                path=str(args.out.resolve()),
                run_script="code/build_me_table.py",
            )
        ),
        target=None
        if args.internal
        else Target(aws=AWSS3Target(bucket=SCRATCH_BUCKET, prefix=f"{SCRATCH_PREFIX}/{name}/")),
    )
    print(json.dumps(params.to_dict(), indent=2))
    if args.dry_run:
        return

    if not args.domain:
        raise SystemExit("No Code Ocean URL: set CO_DOMAIN or pass --domain")
    client = CodeOcean(domain=args.domain, token=os.environ["API_SECRET"])
    asset = client.data_assets.create_data_asset(params)
    print(f"Created {asset.id} ({asset.name}); waiting for it to be ready")
    while asset.state not in (DataAssetState.Ready, DataAssetState.Failed):
        time.sleep(30)
        asset = client.data_assets.get_data_asset(asset.id)
        print(f"  {asset.state.value}", flush=True)
    if asset.state == DataAssetState.Failed:
        raise SystemExit(f"Capture failed: {asset.id}")
    print(f"Ready: {asset.id} {asset.name}")


if __name__ == "__main__":
    main()
