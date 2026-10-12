"""
Temporal Consistency Analyzer Module for Deepfake Forensic Prototype.
Measures inter-frame stability across consecutively sampled video frames.
Calculates Mean Absolute Difference (MAD), Normalized Frame Difference (NFD),
and Root Mean Squared Difference (RMSD).
"""

from typing import List, Dict, Any
import numpy as np


def compute_temporal_differences(
    sampled_frames: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Computes inter-frame temporal variation between adjacent sampled frames.
    
    Mathematical Formulations:
        Let I_t and I_{t-1} denote normalized grayscale frame intensities in [0, 1]
        of spatial dimension H x W.
        
        1. Mean Absolute Difference (MAD):
            MAD(t) = (1 / (H * W)) * sum_{i, j} |I_t(i, j) - I_{t-1}(i, j)|
            Measures average per-pixel intensity fluctuation. Range: [0, 1].
            
        2. Normalized Frame Difference (NFD):
            NFD(t) = sum_{i,j} |I_t - I_{t-1}| / (sum_{i,j} I_t + sum_{i,j} I_{t-1} + eps)
            Relative variation normalized by overall scene luminance. Range: [0, 1].
            
        3. Root Mean Squared Difference (RMSD):
            RMSD(t) = sqrt( (1 / (H * W)) * sum_{i,j} (I_t(i, j) - I_{t-1}(i, j))^2 )
            Penalizes localized abrupt changes more heavily than MAD. Range: [0, 1].
            
    Confounding Factors & Forensic Caveats:
        - High inter-frame differences naturally occur due to rapid camera pan/tilt,
          quick facial head turns, sudden illumination flickers, or video shot transitions.
        - Low differences occur during static freeze frames or talking-head interviews.
        - Therefore, temporal metrics alone CANNOT establish video authenticity or manipulation.
    
    Returns:
        List of dicts containing transition records:
            - 'from_frame_idx': int
            - 'to_frame_idx': int
            - 'timestamp_sec': float (target frame timestamp)
            - 'time_delta_sec': float
            - 'mad': float (Mean Absolute Difference in [0, 1])
            - 'nfd': float (Normalized Frame Difference in [0, 1])
            - 'rmsd': float (Root Mean Squared Difference in [0, 1])
    """
    if len(sampled_frames) < 2:
        return []

    records = []
    
    for i in range(1, len(sampled_frames)):
        prev_item = sampled_frames[i - 1]
        curr_item = sampled_frames[i]
        
        prev_gray = prev_item["gray_image"].astype(np.float64) / 255.0
        curr_gray = curr_item["gray_image"].astype(np.float64) / 255.0
        
        # Verify dimension compatibility
        if prev_gray.shape != curr_gray.shape:
            raise ValueError(
                f"Frame dimension mismatch between frame {prev_item['frame_idx']} "
                f"and frame {curr_item['frame_idx']}."
            )
            
        diff = np.abs(curr_gray - prev_gray)
        mad = float(np.mean(diff))
        
        eps = 1e-12
        sum_abs_diff = float(np.sum(diff))
        sum_total = float(np.sum(curr_gray) + np.sum(prev_gray))
        nfd = float(sum_abs_diff / (sum_total + eps))
        
        rmsd = float(np.sqrt(np.mean((curr_gray - prev_gray) ** 2)))
        
        t_delta = round(curr_item["timestamp_sec"] - prev_item["timestamp_sec"], 3)
        
        records.append({
            "from_frame_idx": prev_item["frame_idx"],
            "to_frame_idx": curr_item["frame_idx"],
            "timestamp_sec": curr_item["timestamp_sec"],
            "time_delta_sec": t_delta,
            "mad": round(mad, 5),
            "nfd": round(nfd, 5),
            "rmsd": round(rmsd, 5),
        })
        
    return records


def aggregate_temporal_summary(temporal_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes summary statistics over all computed frame-difference intervals.
    """
    if not temporal_records:
        return {
            "mean_mad": 0.0,
            "max_mad": 0.0,
            "std_mad": 0.0,
            "mean_nfd": 0.0,
            "mean_rmsd": 0.0
        }
        
    mads = [r["mad"] for r in temporal_records]
    nfds = [r["nfd"] for r in temporal_records]
    rmsds = [r["rmsd"] for r in temporal_records]
    
    return {
        "mean_mad": round(float(np.mean(mads)), 5),
        "max_mad": round(float(np.max(mads)), 5),
        "std_mad": round(float(np.std(mads)), 5),
        "mean_nfd": round(float(np.mean(nfds)), 5),
        "mean_rmsd": round(float(np.mean(rmsds)), 5),
    }
