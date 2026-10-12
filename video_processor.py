"""
Video Processing Module for Deepfake Forensic Prototype.
Handles video validation, metadata inspection, and memory-efficient frame extraction.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
# pyrefly: ignore [missing-import]
import cv2
import numpy as np


def validate_video_file(
    file_path: str,
    max_size_mb: float = 100.0,
    max_duration_sec: float = 120.0
) -> Tuple[bool, Optional[str]]:
    """
    Validates that a video file exists, is non-empty, within size/duration bounds,
    and is decodable by OpenCV.
    
    Returns:
        (is_valid, error_message): Tuple where error_message is None if valid.
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    if file_size_bytes == 0:
        return False, "Uploaded file is 0 bytes (empty file)."
    
    if file_size_mb > max_size_mb:
        return False, (
            f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed limit "
            f"({max_size_mb:.2f} MB)."
        )
    
    # Verify readability using OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        return False, (
            "Cannot open video stream. File may be corrupt, incomplete, "
            "or encoded in an unsupported video container."
        )
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return False, "Video contains invalid metadata (FPS <= 0 or zero frames)."
    
    duration = total_frames / fps
    if duration > max_duration_sec:
        cap.release()
        return False, (
            f"Video duration ({duration:.1f}s) exceeds the prototype limit "
            f"of {max_duration_sec:.1f}s."
        )
    
    # Read first frame as a sanity probe
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return False, "Failed to decode the initial video frame."
    
    return True, None


def get_video_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts essential video container metadata.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to read video file: {file_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Decode FourCC codec string
    codec_chars = [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
    codec = "".join(c for c in codec_chars if c.isprintable()).strip()
    if not codec:
        codec = "Unknown/DirectShow"
    
    duration_sec = total_frames / fps if fps > 0 else 0.0
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    cap.release()
    
    return {
        "filename": os.path.basename(file_path),
        "file_size_mb": round(file_size_mb, 2),
        "resolution": (width, height),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "codec": codec,
    }


def extract_sampled_frames(
    file_path: str,
    num_frames: int = 10,
    target_size: Optional[Tuple[int, int]] = (256, 256)
) -> List[Dict[str, Any]]:
    """
    Extracts evenly spaced frames from the video.
    Frames are immediately resized to target_size to prevent retaining
    full-resolution images in RAM.
    
    Returns:
        List of dicts: [
            {
                "frame_idx": int,
                "timestamp_sec": float,
                "rgb_image": np.ndarray (H, W, 3) in uint8 [0..255] RGB,
                "gray_image": np.ndarray (H, W) in uint8 [0..255] Grayscale
            }, ...
        ]
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {file_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise ValueError("Invalid video stream properties (0 frames or 0 FPS).")
    
    # Determine evenly spaced frame indices
    num_frames = max(2, min(num_frames, total_frames))
    sample_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    extracted = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        
        if not ret or frame_bgr is None:
            continue
        
        # Maintain aspect ratio for clean visual inspection and accurate face detection
        orig_h, orig_w = frame_bgr.shape[:2]
        max_disp_dim = 720
        if max(orig_h, orig_w) > max_disp_dim:
            scale = max_disp_dim / float(max(orig_h, orig_w))
            disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
            frame_disp_bgr = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            frame_disp_bgr = frame_bgr
        
        frame_rgb = cv2.cvtColor(frame_disp_bgr, cv2.COLOR_BGR2RGB)
        
        # Standardized dimensions for frequency 2D FFT and temporal analysis
        if target_size is not None:
            frame_square_bgr = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
        else:
            frame_square_bgr = frame_disp_bgr
        frame_gray = cv2.cvtColor(frame_square_bgr, cv2.COLOR_BGR2GRAY)
        
        timestamp_sec = float(idx / fps)
        
        extracted.append({
            "frame_idx": int(idx),
            "timestamp_sec": round(timestamp_sec, 3),
            "rgb_image": frame_rgb,
            "gray_image": frame_gray,
        })
    
    cap.release()
    
    if not extracted:
        raise RuntimeError("No frames could be extracted from the video source.")
    
    return extracted


def create_synthetic_test_video(
    output_path: str,
    num_frames: int = 30,
    fps: int = 15,
    width: int = 256,
    height: int = 256,
    simulate_artifact: bool = False
) -> str:
    """
    Generates a lightweight synthetic MP4 video for local testing and reviewer demonstrations.
    Includes a stylized face-like shape with continuous motion and an optional high-frequency artifact.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Base canvas
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving gradient background
        shift = int((i / num_frames) * 60)
        img[:, :] = (30 + shift, 25, 45)
        
        # Simulated face (head circle)
        cx = width // 2 + int(np.sin(i * 0.2) * 20)
        cy = height // 2 + int(np.cos(i * 0.2) * 15)
        cv2.circle(img, (cx, cy), 50, (235, 195, 175), -1) # Skin tone
        
        # Eyes
        cv2.circle(img, (cx - 18, cy - 10), 6, (40, 40, 40), -1)
        cv2.circle(img, (cx + 18, cy - 10), 6, (40, 40, 40), -1)
        
        # Mouth
        cv2.ellipse(img, (cx, cy + 20), (16, 8), 0, 0, 180, (180, 50, 50), -1)
        
        # Optional high-frequency checkerboard pattern to simulate deepfake blending artifact
        if simulate_artifact and (i % 2 == 1):
            patch = (np.indices((30, 30)).sum(axis=0) % 2) * 180
            patch_rgb = np.stack([patch]*3, axis=-1).astype(np.uint8)
            img[cy-15:cy+15, cx-15:cx+15] = cv2.addWeighted(
                img[cy-15:cy+15, cx-15:cx+15], 0.4, patch_rgb, 0.6, 0
            )
            
        out.write(img)
        
    out.release()
    return output_path
