from pathlib import Path
import json
import datetime

from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import run_batch_analysis
from build_all_tongue_movements import build_all_tongue_movements

save_root = "/root/capsule/scratch/session_analysis_fall2026"
data_root = Path("/root/capsule/data")

# Load list from JSON
pred_list_path = Path("/root/capsule/scratch/pred_csv_list_20250113.json")
with open(pred_list_path, "r") as f:
    pred_csv_list = json.load(f)

run_batch_analysis(pred_csv_list, data_root, save_root)

# pool into a freshly dated parquet
date_tag = datetime.date.today().strftime("%m%d%Y")
out = Path(f"/root/capsule/scratch/temp/all_tongue_movements_{date_tag}.parquet")
build_all_tongue_movements(base=Path(save_root), out=out)
