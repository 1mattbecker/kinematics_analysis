"""Reproducible-run entry point: the library's tongue-kinematics batch over the session list, then
the pooled tables.

Each session goes to ``kin_utils.KINEMATICS_ROOT/<session>/`` (library ``run_batch``; finished
sessions are skipped unless ``FORCE``). Trial-check errors from data-utils ("Reward before choice
time", 2025 sessions) are turned into the ``trials.reward_before_choice`` flag instead of stopping
the session. ``build_all_tongue_movements`` then pools the sessions that pass the quality filter.
"""
import json
from pathlib import Path

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tasks import dynamic_foraging_tongue as dft

from build_all_tongue_movements import build_all_tongue_movements
from kin_utils import KINEMATICS_ROOT

DATA_ROOT = Path("/root/capsule/data")
POSE_LIST = Path("/root/capsule/scratch/pred_csv_list_20250113.json")
FORCE = False   # True reprocesses finished sessions (needed after a library version change)

if __name__ == "__main__":
    pose_paths = list(dict.fromkeys(json.loads(POSE_LIST.read_text())))
    status = dft.run_batch(pose_paths, DATA_ROOT, KINEMATICS_ROOT,
                           ignore_trial_errors=True, force=FORCE)
    print(status.to_string())
    build_all_tongue_movements()
