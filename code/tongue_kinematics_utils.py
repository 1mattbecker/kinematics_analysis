import numpy as np
import pandas as pd


def filter_timestamps_refractory(timestamps, t_refractory):
    
    # Sort the timestamps
    timestamps.sort()
    
    filtered_timestamps = []
    last_timestamp = None
    
    for ts in timestamps:
        if last_timestamp is None or (ts - last_timestamp) > t_refractory:
            filtered_timestamps.append(ts)
            last_timestamp = ts
    
    print(f"Filtered {len(timestamps)-len(filtered_timestamps)} events!")

    return filtered_timestamps


def calculate_metrics_witheventkeys(ground_truth, detected_events, time_window=0.05):
    # calculate metrics, output include eventkeys for plotting
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    gt_events = np.array(ground_truth)
    detected = np.array(detected_events)
    
    # Sort events for easier comparison
    gt_events = np.sort(gt_events)
    detected = np.sort(detected)
    
    gt_index = 0
    det_index = 0
    
    # Dictionaries to store event keys
    gt_keys = {event: 'Unclassified' for event in gt_events}
    det_keys = {event: 'Unclassified' for event in detected}
    
    while gt_index < len(gt_events) and det_index < len(detected):
        if abs(detected[det_index] - gt_events[gt_index]) <= time_window:
            tp += 1
            gt_keys[gt_events[gt_index]] = 'True Positive'
            det_keys[detected[det_index]] = 'True Positive'
            gt_index += 1
            det_index += 1
        elif detected[det_index] < gt_events[gt_index]:
            fp += 1
            det_keys[detected[det_index]] = 'False Positive'
            det_index += 1
        else:
            fn += 1
            gt_keys[gt_events[gt_index]] = 'False Negative'
            gt_index += 1
    
    # Remaining false positives
    while det_index < len(detected):
        fp += 1
        det_keys[detected[det_index]] = 'False Positive'
        det_index += 1
    
    # Remaining false negatives
    while gt_index < len(gt_events):
        fn += 1
        gt_keys[gt_events[gt_index]] = 'False Negative'
        gt_index += 1
    
    # Assuming we have a defined observation period
    total_observations = max(gt_events[-1] if gt_events.size else 0,
                             detected[-1] if detected.size else 0)
    tn = total_observations - (tp + fp + fn)

    gt_df = pd.DataFrame(list(gt_keys.items()), columns=['Time', 'Status'])
    det_df = pd.DataFrame(list(det_keys.items()), columns=['Time', 'Status'])

    
    return tp, fp, fn, tn, gt_df, det_df


def calculate_metrics(ground_truth, detected_events, time_window=0.05):
    # calculate sensitivity / specificity
    # detect concurrent licks with 50 msec shoulders

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    gt_events = np.array(ground_truth)
    detected = np.array(detected_events)
    
    # Sort events (likely already sorted)
    gt_events = np.sort(gt_events)
    detected = np.sort(detected)
    
    gt_index = 0
    det_index = 0
    
    while gt_index < len(gt_events) and det_index < len(detected):
        if abs(detected[det_index] - gt_events[gt_index]) <= time_window:
            tp += 1
            gt_index += 1
            det_index += 1
        elif detected[det_index] < gt_events[gt_index]:
            fp += 1
            det_index += 1
        else:
            fn += 1
            gt_index += 1
    
    # Count remaining false positives
    fp += len(detected) - det_index
    
    # Count remaining false negatives
    fn += len(gt_events) - gt_index
    
    # Assuming we have a defined observation period
    total_observations = max(gt_events[-1] if gt_events.size else 0,
                             detected[-1] if detected.size else 0)
    tn = total_observations - (tp + fp + fn)
    
    return tp, fp, fn, tn


def detect_licks(tongue_df, timestamps, spoutL, spoutR, threshold):
    """
    Detect the timestamps of licks based on proximity to spouts.

    Parameters:
    - tongue_df: Pandas DataFrame with columns 'x' and 'y' for tongue positions
    - timestamps: Pandas Series with timestamps corresponding to tongue_df
    - spoutL: Pandas Series with x and y coordinates of the left spout
    - spoutR: Pandas Series with x and y coordinates of the right spout
    - threshold: Distance threshold for detecting a lick

    Returns:
    - List of timestamps for detected licks
    """

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    detected_licks = []
    is_licking = False

    # Convert spout positions to tuples
    spoutL_pos = (spoutL['x'], spoutL['y'])
    spoutR_pos = (spoutR['x'], spoutR['y'])

    for i in range(len(tongue_df)):
        # Extract tongue position
        tongue_pos = tongue_df.iloc[i]
        
        # Skip rows where tongue position is NaN
        if pd.isna(tongue_pos['x']) or pd.isna(tongue_pos['y']):
            continue
        
        tongue_pos = (tongue_pos['x'], tongue_pos['y'])
        
        dist_to_spoutL = distance(tongue_pos, spoutL_pos)
        dist_to_spoutR = distance(tongue_pos, spoutR_pos)

        if (dist_to_spoutL <= threshold or dist_to_spoutR <= threshold):
            if not is_licking:
                # Start of a lick
                detected_licks.append(timestamps.iloc[i])
                is_licking = True
        else:
            is_licking = False

    return detected_licks


def mask_keypoint_data(keypoint_dfs,keypoint, confidence_threshold=0.9, mask_value=np.nan):
    """
    Mask the 'x' or 'y' data for a specific keypoint based on a confidence threshold.

    Parameters:
    - keypoint_dfs: keypoint dataframe from 'load_keypoints_from_csv'
    - keypoint: str, name of the keypoint to process
    - confidence_threshold: float, the confidence value threshold for masking
    - mask_value: value to use for masking (default is np.nan)

    Returns:
    - masked_df: DataFrame with masked 'x' and 'y' values
    """
    if keypoint in keypoint_dfs:
        kp_df = keypoint_dfs[keypoint].copy()  # Copy to avoid modifying original DataFrame
        
        # Apply the mask based on the confidence threshold
        kp_df.loc[kp_df['confidence'] < confidence_threshold, ['x', 'y']] = mask_value
        
        return kp_df
    else:
        print(f"Keypoint {keypoint} not found")
        return None

def integrate_keypoints_with_video_time(video_csv_path, keypoint_dfs):
    """
    Imports, checks, and preprocesses video CSV, then trims keypoint data to match video length.

    Parameters:
    - video_csv_path: Path to the original bonsai video acquisition CSV
    - keypoint_dfs: Dictionary of dataframes from load_keypoints_from_csv

    Returns:
    - keypoint_dfs_trimmed: Trimmed keypoint dataframes
    - video_csv_trimmed: Processed and trimmed video CSV dataframe
    - keypoint_timebase: Timebase for kinematics data, in time aligned to NWB time.
    """

    # Step 1: Load video CSV
    video_csv = pd.read_csv(video_csv_path, names=['Behav_Time', 'Frame', 'Camera_Time', 'Gain', 'Exposure'])
    
    # Step 2: Convert Camera_Time to seconds
    video_csv['Camera_Time'] = video_csv['Camera_Time'] / 1e9

    # Step 3: Quality control checks
    def check_frame_monotonicity(df):
        """Ensure frame numbers increase strictly by 1."""
        frame_diff = df['Frame'].diff().dropna()
        if not (frame_diff == 1).all():
            print("Warning: Non-monotonic frame numbering detected.")
            print(df.loc[frame_diff[frame_diff != 1].index])
        else:
            print("Video QC: Frame numbers are sequential with no gaps.")

    def check_timing_consistency(df, expected_interval=1/500):
        """Check consistency between Behav_Time and Camera_Time."""
        behav_diffs = df['Behav_Time'].diff().dropna()
        camera_diffs = df['Camera_Time'].diff().dropna()
        time_diff = (behav_diffs - camera_diffs).abs()
        flagged_indices = time_diff[time_diff > expected_interval * 2].index

        if not flagged_indices.empty:
            print("Warning: Timing differences exceed expected variation.")
            print(df.loc[flagged_indices, ['Behav_Time', 'Camera_Time']])
        else:
            print("Video QC: Timing differences are within expected range.")

    check_frame_monotonicity(video_csv)
    check_timing_consistency(video_csv)

    # Step 4: Trim kinematics timebase to match video
    def trim_kinematics_timebase_to_match(keypoint_dfs, video_csv):
        LP_samples = len(keypoint_dfs[list(keypoint_dfs.keys())[0]])
        video_samples = len(video_csv)

        if LP_samples > video_samples:
            print(f"keypoint_df trimmed from {LP_samples} to {video_samples}")
        elif LP_samples < video_samples:
            print(f"video_csv trimmed from {video_samples} to {LP_samples}")
        else:
            print("no change")

        min_samples = min(LP_samples, video_samples)
        video_csv_trimmed = video_csv.iloc[:min_samples]

        keypoint_dfs_trimmed = keypoint_dfs.copy()
        for key in keypoint_dfs.keys():
            keypoint_dfs_trimmed[key] = keypoint_dfs[key].iloc[:min_samples]

        return keypoint_dfs_trimmed, video_csv_trimmed

    keypoint_dfs_trimmed, video_csv_trimmed = trim_kinematics_timebase_to_match(keypoint_dfs, video_csv)
    keypoint_timebase = video_csv_trimmed['Behav_Time']

    # Step 5: Add 'time' column to each keypoint dataframe
    for key in keypoint_dfs_trimmed.keys():
        keypoint_dfs_trimmed[key].insert(0, 'time', keypoint_timebase - keypoint_timebase.iloc[0])

    return keypoint_dfs_trimmed, video_csv_trimmed, keypoint_timebase

def load_keypoints_from_csv(path_to_csv):
    """
    Load keypoints from Lightning Pose csv into data frame

    Parameters:
    - path_to_csv: path to your csv file from LP. Assumes follow format:
        -first column is extraneous
        -first row is keypoint labels (e.g. 'tongue_tip')
        -second row is data content labels (e.g. 'x' position)

    Returns:
    - keypoint_dfs: DataFrame with 'x', 'y', and 'confidence' values, organized by keypoint
    """
    
    df = pd.read_csv(path_to_csv, dtype=str, low_memory=False)

    #remove first column
    df = df.iloc[:, 1:]

    # Extract header information
    header_labels = df.iloc[0]  # First row: keypoint labels
    types = df.iloc[1]          # Second row: 'x', 'y', or 'confidence'

    # Drop the first two rows and reset the index
    df = df[2:].reset_index(drop=True)

    # Convert data to numeric, replacing errors with NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Create a dictionary to store DataFrames for each keypoint
    keypoint_dfs = {}

    # Loop over the columns in the DataFrame
    for i in range(0, len(df.columns), 3):
        # Extract keypoint name from the header_labels
        keypoint = header_labels.iloc[i]
        
        # Check if the columns exist in the DataFrame
        if i + 2 < len(df.columns):
            keypoint_df = pd.DataFrame({
                'x': df.iloc[:, i].astype('float'),
                'y': df.iloc[:, i + 1].astype('float'),
                'confidence': df.iloc[:, i + 2].astype('float')
            })
            keypoint_dfs[keypoint] = keypoint_df
        else:
            print(f"Warning: Missing columns for keypoint {keypoint}")
    
    print(f'keypoints extracted: {list(keypoint_dfs.keys())}')

    return keypoint_dfs