from pathlib import Path
import json
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import run_batch_analysis

save_root = "/root/capsule/scratch/session_analysis_out_of_distribution_full"
data_root = Path("/root/capsule/data")

# Load list from JSON
pred_list_path = Path("/root/capsule/scratch/pred_csv_list.json")
with open(pred_list_path, "r") as f:
    pred_csv_list = json.load(f)

run_batch_analysis(pred_csv_list, data_root, save_root)
