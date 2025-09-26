from pathlib import Path
from aind_dynamic_foraging_behavior_video_analysis.kinematics.tongue_analysis import run_batch_analysis
from scratch.pred_csv_list import pred_csv_list

save_root = "/root/capsule/scratch/session_analysis_out_of_distribution_full"
data_root = Path("/root/capsule/data")

test_list = [pred_csv_list[0]]

run_batch_analysis(pred_csv_list, data_root, save_root)
