from pathlib import Path
import json
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import run_batch_analysis

# Base paths
data_root = Path("/root/capsule/data")
scratch_root = Path("/root/capsule/scratch")

# Model keys and corresponding JSON list files
model_keys = [
    "ephys_bottom-camera",
    "ephys_BottomCamera_Pylon2Test1",
    "ephys_BottomCamera_Pylon2Test2",
]

for key in model_keys:
    pred_list_path = scratch_root / f"pred_csv_list__{key}.json"
    if not pred_list_path.exists():
        print(f"⚠️ Missing list for {key}: {pred_list_path}")
        continue

    with open(pred_list_path, "r") as f:
        pred_csv_list = json.load(f)

    # Create save folder named after model
    save_root = Path(f"/root/capsule/scratch/session_analysis_{key}")
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running batch for {key} ===")
    print(f" - N files: {len(pred_csv_list)}")
    print(f" - Save to: {save_root}")
    run_batch_analysis(pred_csv_list, data_root, save_root)
