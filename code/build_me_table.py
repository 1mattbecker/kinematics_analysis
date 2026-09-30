"""
build_me_table.py — aligned motion-energy table for the FIP sessions.

For every ``used_in_fip05_07`` session and each camera, writes per-frame
corrected Harp time plus the cleaned motion-energy trace, so the FIP notebooks
attach one data asset instead of ~200 raw-behavior and ME assets. The spec is
``code/fip_me_aligned_table_plan.md``.

Timing comes from the library's ``video_timing_qc`` (nothing is reimplemented
here): the Harp trigger log when the session has one, the camera CSV alone
otherwise. A log that is present but rejected refuses the camera; there is no
fallback to the CSV. Refused cameras get an ``index.csv`` row and no file, and
one bad camera never stops the build.

Output (``--out``)::

    index.csv                        one row per session x camera
    build_info.json                  library commit, build date, ME run ids
    <session>/BottomCamera.parquet   row i is video frame i
    <session>/SideCameraRight.parquet

Inputs are read over anonymous HTTPS: raw sessions from ``aind-open-data``,
ME results from ``aind-scratch-data`` (folders named in
``metadata/me_assets_fip.csv``). Only CSVs, trigger logs, ME metadata and
``.npy`` files are read, never videos.

Usage (from ``code/``)::

    python build_me_table.py --dry-run                     # timing only, all sessions
    python build_me_table.py --sessions 816214_2025-12-02  # subset
    python build_me_table.py --workers 4                   # full build
    python build_me_table.py --force                       # rebuild built sessions
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import aind_dynamic_foraging_behavior_video_analysis as video_analysis
from aind_dynamic_foraging_behavior_video_analysis import video_timing_qc as vtq

CODE_DIR = Path(__file__).resolve().parent
REPO = CODE_DIR.parent
SESSIONS_CSV = REPO / "metadata" / "me_sessions_fip_curated.csv"
ME_ASSETS_CSV = REPO / "metadata" / "me_assets_fip.csv"
DEFAULT_OUT = REPO / "results" / "fip_motion_energy_aligned"
DEFAULT_REPORT = REPO / "results" / "build_me_table_dry_run.csv"

OPEN_DATA_URL = "https://aind-open-data.s3.amazonaws.com"
SCRATCH_URL = "https://aind-scratch-data.s3.amazonaws.com"
ME_PREFIX = "matt.becker/motion_energy"
# One per session, shared by every camera (Harp Behavior register 94)
TRIGGER_LOG = "behavior/raw.harp/BehaviorEvents/Event_94.bin"

# Output camera name -> name in the raw asset (and ME files), per layout
CAMERAS = {
    "BottomCamera": {"flat": "bottom_camera", "nested": "BottomCamera"},
    "SideCameraRight": {"flat": "side_camera_right", "nested": "SideCameraRight"},
}
HARP_SOURCES = [
    "trigger_log",
    "original",
    "reindexed",
    "estimated_camera_fit",
    "glitch_interpolated",
]
INDEX_COLUMNS = [
    "session", "subject", "camera", "source_camera", "layout",
    "raw_asset_id", "me_asset_id", "source_video",
    "n_frames", "frame_interval_s", "harp_start", "harp_end",
    "timing_source", "action", "status", "error", "failed_checks",
    "frames_lost", "glitch_rows", "clock_rate_ppm", "video_frame_count_diff",
    *[f"n_{source}" for source in HARP_SOURCES],
    "padded",
]
# Dry-run report only: what the log was checked against
REPORT_COLUMNS = INDEX_COLUMNS + ["n_trigger_events", "n_exposures", "me_asset_name"]


# --- Reading from S3 over HTTPS -------------------------------------------


def _urlopen(url, tries=4):
    """Open ``url``, retrying transient errors; None on 404."""
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)


def download(url, dest):
    """Save ``url`` to ``dest``; return ``dest``, or None if it does not exist."""
    response = _urlopen(url)
    if response is None:
        return None
    with response, open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    return dest


def read_json(url):
    """Read a JSON object from ``url``; None if it does not exist."""
    response = _urlopen(url)
    if response is None:
        return None
    with response:
        return json.load(response)


def list_keys(bucket_url, prefix):
    """Object keys under ``prefix`` (one listing page is enough here)."""
    query = urllib.parse.urlencode({"list-type": 2, "prefix": prefix})
    response = _urlopen(f"{bucket_url}/?{query}")
    if response is None:
        raise FileNotFoundError(f"Cannot list {bucket_url}/{prefix}")
    with response:
        text = response.read().decode()
    return re.findall(r"<Key>(.*?)</Key>", text)


def detect_layout(raw_session):
    """``flat`` (``bottom_camera.csv``) or ``nested`` (``BottomCamera/metadata.csv``)."""
    keys = set(list_keys(OPEN_DATA_URL, f"{raw_session}/behavior-videos/"))
    found = [
        layout
        for layout in ("flat", "nested")
        if all(f"{raw_session}/{csv_key(layout, CAMERAS[c][layout])}" in keys for c in CAMERAS)
    ]
    if len(found) != 1:
        raise ValueError(f"Camera CSV layout not recognised ({found or 'none found'})")
    return found[0]


def csv_key(layout, source_camera):
    """Path of a camera's timestamp CSV inside the raw asset."""
    if layout == "flat":
        return f"behavior-videos/{source_camera}.csv"
    return f"behavior-videos/{source_camera}/metadata.csv"


# --- Per camera -----------------------------------------------------------


def build_camera(camera, info, work_dir, trigger_times, log_error, out_dir, dry_run):
    """Check, correct and (unless ``dry_run``) write one camera.

    Fills ``info`` (one index row) in place. Raises on any refusal; the caller
    records the error.
    """
    source = info["source_camera"]
    me_url = f"{SCRATCH_URL}/{ME_PREFIX}/{info['me_asset_name']}/{source}"

    csv_path = download(
        f"{OPEN_DATA_URL}/{info['session']}/{csv_key(info['layout'], source)}",
        work_dir / f"{source}.csv",
    )
    if csv_path is None:
        raise FileNotFoundError("Camera CSV missing")
    timing = vtq.load_video_timing(csv_path)
    info["n_frames"] = len(timing)
    info["frame_interval_s"] = vtq.frame_interval(timing["harp_time_raw"])
    info["n_exposures"] = int(timing["frame_number"].iloc[-1] - timing["frame_number"].iloc[0] + 1)

    me_meta = read_json(f"{me_url}_me_metadata.json")
    if me_meta is None:
        raise FileNotFoundError(f"{source}_me_metadata.json missing from the ME asset")
    info["source_video"] = me_meta["video_path"]
    if not me_meta["video_path"].endswith(".mp4"):
        raise ValueError(f"ME computed from {me_meta['video_path']}, not an .mp4")
    if me_meta["start_frame"] is not None or me_meta["end_frame"] is not None:
        raise ValueError(
            f"ME covers frames {me_meta['start_frame']}..{me_meta['end_frame']}, not the whole video"
        )
    n_decoded = int(me_meta["n_frames_decoded"])
    n_me = int(me_meta["n_me_frames"])

    checks = vtq.check_video_timing(timing, video_frame_count=n_decoded)
    by_check = checks.set_index("check")
    info["action"] = vtq.timing_action(checks)
    info["failed_checks"] = ";".join(checks.loc[checks["passed"].eq(False), "check"])
    info["frames_lost"] = int(by_check.loc["no_frames_lost", "count"])
    info["glitch_rows"] = " ".join(map(str, by_check.loc["harp_has_no_glitches", "rows"]))
    info["clock_rate_ppm"] = int(by_check.loc["clock_rates_agree", "count"])
    info["video_frame_count_diff"] = int(by_check.loc["video_frame_count", "count"])

    # timing_action does not look at the video frame count: a mismatch shifts every
    # later frame, and where the frames went is not recorded, so refuse it here
    if not by_check.loc["video_frame_count", "passed"]:
        raise ValueError(f"video_frame_count failed: {by_check.loc['video_frame_count', 'message']}")
    # A log that is present but unreadable is a rejected log, not a missing one
    if log_error is not None:
        raise ValueError(f"Trigger log unreadable: {log_error}")

    fixed = vtq.correct_video_timing(timing, trigger_times=trigger_times)
    info["harp_start"] = float(fixed["harp_time"].iloc[0])
    info["harp_end"] = float(fixed["harp_time"].iloc[-1])
    counts = fixed["harp_source"].value_counts()
    for harp_source in HARP_SOURCES:
        info[f"n_{harp_source}"] = int(counts.get(harp_source, 0))

    # aind-motion-energy differences consecutive frames: one fewer value than frames
    # (none for frame 0). A leading NaN makes row i frame i. Anything else is refused.
    info["padded"] = n_me == n_decoded - 1
    if n_me + info["padded"] != len(timing):
        raise ValueError(f"{n_me} ME values for {len(timing)} CSV rows")

    if dry_run:
        return
    npy_path = download(f"{me_url}_motion_energy_clean.npy", work_dir / f"{source}_me_clean.npy")
    if npy_path is None:
        raise FileNotFoundError(f"{source}_motion_energy_clean.npy missing from the ME asset")
    me = np.load(npy_path).astype("float32")
    if len(me) != n_me:
        raise ValueError(f"ME file has {len(me)} values, its metadata says {n_me}")
    if info["padded"]:
        me = np.concatenate([np.array([np.nan], dtype="float32"), me])

    table = pd.DataFrame(
        {
            "harp_time": fixed["harp_time"].to_numpy("float64"),
            "harp_source": pd.Categorical(fixed["harp_source"]),
            "frame_number": fixed["frame_number"].to_numpy("int64"),
            "camera_time": fixed["camera_time"].to_numpy("float64"),
            "me_clean": me,
        }
    )
    session_dir = out_dir / info["session"]
    session_dir.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted build never leaves a partial file
    tmp_path = session_dir / f".{camera}.parquet.tmp"
    table.to_parquet(tmp_path, index=False, compression="zstd")
    os.replace(tmp_path, session_dir / f"{camera}.parquet")


# --- Per session ----------------------------------------------------------


def build_session(session, out_dir, dry_run, tmp_root=None):
    """Build both cameras of one session; return their index rows.

    Never raises: a failure refuses the camera (or, before the cameras are
    reached, both cameras) with the error text.
    """
    base = {
        "session": session["raw_session"],
        "subject": session["subject"],
        "raw_asset_id": session["raw_asset_id"],
        "me_asset_id": session["me_asset_id"],
        "me_asset_name": session["me_asset_name"],
    }
    rows = []
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        work_dir = Path(tmp)
        try:
            layout = detect_layout(session["raw_session"])
            trigger_times, log_error = None, None
            log_path = download(
                f"{OPEN_DATA_URL}/{session['raw_session']}/{TRIGGER_LOG}", work_dir / "Event_94.bin"
            )
            if log_path is not None:
                try:
                    trigger_times = vtq.read_harp_trigger_log(log_path)
                except ValueError as e:
                    log_error = str(e)
            session_error = None
        except Exception as e:  # noqa: BLE001 — recorded, never raised
            layout, log_path, trigger_times, log_error = None, None, None, None
            session_error = f"{type(e).__name__}: {e}"

        for camera, source_names in CAMERAS.items():
            info = {
                **base,
                "camera": camera,
                "layout": layout,
                "source_camera": source_names.get(layout),
                "timing_source": "trigger_log" if log_path is not None else "csv",
                "n_trigger_events": None if trigger_times is None else len(trigger_times),
                "status": "ok",
                "error": None,
            }
            try:
                if session_error is not None:
                    raise RuntimeError(session_error)
                build_camera(camera, info, work_dir, trigger_times, log_error, out_dir, dry_run)
            except Exception as e:  # noqa: BLE001 — one bad camera never stops the build
                info["status"] = "refused"
                info["error"] = str(e) if session_error else f"{type(e).__name__}: {e}"
                if not isinstance(e, (ValueError, FileNotFoundError, RuntimeError)):
                    info["error"] += " | " + traceback.format_exc(limit=3).replace("\n", " ")
            rows.append(info)
    return rows


# --- Build ----------------------------------------------------------------


def load_sessions(selected=None):
    """The ``used_in_fip05_07`` sessions joined to their ME assets."""
    sessions = pd.read_csv(SESSIONS_CSV)
    sessions = sessions.loc[sessions["used_in_fip05_07"], ["raw_session", "subject"]]
    assets = pd.read_csv(ME_ASSETS_CSV)
    merged = sessions.merge(assets, on="raw_session", how="left", validate="1:1")
    if merged["me_asset_name"].isna().any():
        missing = merged.loc[merged["me_asset_name"].isna(), "raw_session"].tolist()
        raise ValueError(f"No ME asset in {ME_ASSETS_CSV.name} for {missing}")
    if selected:
        # Accept full raw names or any leading part after "behavior_"
        keep = merged["raw_session"].map(
            lambda name: any(name == s or name.startswith(f"behavior_{s}") for s in selected)
        )
        unmatched = [
            s
            for s in selected
            if not any(n == s or n.startswith(f"behavior_{s}") for n in merged["raw_session"])
        ]
        if unmatched:
            raise ValueError(f"--sessions matched nothing: {unmatched}")
        merged = merged[keep]
    return merged.to_dict("records")


def is_built(index, out_dir, session):
    """Both cameras indexed, and every ``ok`` camera's file present."""
    rows = index[index["session"] == session]
    if set(rows["camera"]) != set(CAMERAS):
        return False
    ok = rows[rows["status"] == "ok"]
    return all((out_dir / session / f"{camera}.parquet").exists() for camera in ok["camera"])


def write_csv(frame, path):
    """Write a CSV atomically."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def library_commit():
    """Commit of the installed ``video_timing_qc`` library, if it can be found."""
    try:
        direct_url = importlib.metadata.distribution(
            "aind-dynamic-foraging-behavior-video-analysis"
        ).read_text("direct_url.json")
        commit = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        return subprocess.run(
            ["git", "-C", str(Path(vtq.__file__).parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def repo_commit():
    """Commit of this repo, with ``-dirty`` if it has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def summarize(rows):
    """Print counts per action and status, and every refusal."""
    table = pd.DataFrame(rows)
    print(f"\n{len(table)} cameras, {table['session'].nunique()} sessions")
    print(table.groupby(["action", "status"], dropna=False).size().to_string())
    print(table["timing_source"].value_counts().to_string())
    refused = table[table["status"] == "refused"]
    print(f"\nRefused: {len(refused)} cameras")
    for _, row in refused.iterrows():
        print(f"  {row['session']} {row['camera']}: {row['error']}")


def main():
    """Parse arguments and build (or dry-run) the table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sessions", nargs="+", help="Raw names or leading parts (816214_2025-12-02)")
    parser.add_argument("--dry-run", action="store_true", help="Timing only: no ME, no writes but --report")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Dry-run report CSV")
    parser.add_argument("--force", action="store_true", help="Rebuild sessions already built")
    parser.add_argument("--workers", type=int, default=1, help="Sessions built in parallel")
    parser.add_argument("--tmp", type=Path, help="Scratch folder for downloads")
    args = parser.parse_args()

    sessions = load_sessions(args.sessions)
    index_path = args.out / "index.csv"
    index = pd.DataFrame(columns=INDEX_COLUMNS)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
            index = pd.read_csv(index_path)
        if not args.force:
            built = [s for s in sessions if is_built(index, args.out, s["raw_session"])]
            if built:
                print(f"Skipping {len(built)} sessions already built (--force to rebuild)")
            sessions = [s for s in sessions if s not in built]
    print(f"{'Dry run' if args.dry_run else 'Building'}: {len(sessions)} sessions")

    all_rows = []

    def record(rows):
        """Keep a session's rows and save progress, so an interrupted build keeps them."""
        nonlocal index
        all_rows.extend(rows)
        for row in rows:
            status = "ok" if row["status"] == "ok" else f"REFUSED {row['error'][:120]}"
            print(f"  {row['session']} {row['camera']}: {row.get('action')} | {status}", flush=True)
        if args.dry_run:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_csv(pd.DataFrame(all_rows).reindex(columns=REPORT_COLUMNS), args.report)
        else:
            new = pd.DataFrame(rows).reindex(columns=INDEX_COLUMNS)
            index = pd.concat([index[index["session"] != rows[0]["session"]], new])
            write_csv(index.sort_values(["session", "camera"]), index_path)

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [
                pool.submit(build_session, s, args.out, args.dry_run, args.tmp) for s in sessions
            ]
            for future in as_completed(futures):
                record(future.result())
    else:
        for s in sessions:
            record(build_session(s, args.out, args.dry_run, args.tmp))

    if not args.dry_run:
        assets = pd.read_csv(ME_ASSETS_CSV)
        build_info = {
            "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "library": "aind-dynamic-foraging-behavior-video-analysis",
            "library_version": video_analysis.__version__,
            "library_commit": library_commit(),
            "repo_commit": repo_commit(),
            "me_run_ids": sorted(assets["me_run_id"].unique()),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sessions_in_index": int(index["session"].nunique()),
            "cameras_ok": int((index["status"] == "ok").sum()),
            "cameras_refused": int((index["status"] == "refused").sum()),
        }
        (args.out / "build_info.json").write_text(json.dumps(build_info, indent=2) + "\n")
    if all_rows:
        summarize(all_rows)


if __name__ == "__main__":
    main()
