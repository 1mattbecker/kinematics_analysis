
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
    
    df = pd.read_csv(path_to_csv)

    #remove first column
    df = df.iloc[:, 1:]

    # Extract header information
    header_labels = df.iloc[0]  # First row: keypoint labels
    types = df.iloc[1]          # Second row: 'x', 'y', or 'confidence'

    # Drop the first two rows and reset the index
    df = df[2:].reset_index(drop=True)

    # Create a dictionary to store DataFrames for each keypoint
    keypoint_dfs = {}

    # Loop over the columns in the DataFrame
    for i in range(0, len(df.columns), 3):
        # Extract keypoint name from the header_labels
        keypoint = header_labels.iloc[i]
        
        # Define column names for this keypoint
        x_col = header_labels.iloc[i] + '_x'
        y_col = header_labels.iloc[i] + '_y'
        conf_col = header_labels.iloc[i] + '_confidence'
        
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
    return keypoint_dfs