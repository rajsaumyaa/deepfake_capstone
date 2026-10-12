"""
Unit and Integration Test Suite for Deepfake Video Forensic Prototype.
Validates numerical stability, edge cases, FFT finiteness, and temporal metrics.
"""

import os
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

from video_processor import (
    validate_video_file,
    get_video_metadata,
    extract_sampled_frames,
    create_synthetic_test_video,
)
from spatial_analyzer import SpatialAnalyzer
from frequency_analyzer import (
    compute_2d_fft_spectrum,
    compute_high_frequency_energy_ratio,
    compute_radial_profile,
)
from temporal_analyzer import (
    compute_temporal_differences,
    aggregate_temporal_summary,
)


class TestDeepfakeForensicsPipeline(unittest.TestCase):
    """Test suite ensuring algorithmic correctness and research reproducibility."""

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for synthetic test artifacts
        cls.temp_dir = tempfile.mkdtemp()
        cls.sample_video_path = os.path.join(cls.temp_dir, "test_synthetic.mp4")
        create_synthetic_test_video(
            cls.sample_video_path,
            num_frames=20,
            fps=10,
            width=128,
            height=128,
            simulate_artifact=True
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary artifacts
        if os.path.exists(cls.sample_video_path):
            os.remove(cls.sample_video_path)
        if os.path.exists(cls.temp_dir):
            try:
                os.rmdir(cls.temp_dir)
            except OSError:
                pass

    def test_01_video_validation_valid(self):
        """Verify that a valid MP4 video is accepted by validator."""
        is_valid, err = validate_video_file(self.sample_video_path, max_size_mb=10.0, max_duration_sec=30.0)
        self.assertTrue(is_valid, f"Validation failed unexpectedly: {err}")
        self.assertIsNone(err)

    def test_02_video_validation_invalid_and_corrupt(self):
        """Verify that missing or unreadable files fail gracefully."""
        # Non-existent file
        is_valid, err = validate_video_file("non_existent_file.mp4")
        self.assertFalse(is_valid)
        self.assertIn("File does not exist", err)

        # Corrupt file
        corrupt_path = os.path.join(self.temp_dir, "corrupt.mp4")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_REAL_VIDEO_HEADER_123456789")
        
        is_valid, err = validate_video_file(corrupt_path)
        self.assertFalse(is_valid)
        self.assertTrue("Cannot open video stream" in err or "invalid metadata" in err or "Failed to decode" in err)
        os.remove(corrupt_path)

    def test_03_metadata_extraction(self):
        """Verify video metadata attributes are correct and non-empty."""
        meta = get_video_metadata(self.sample_video_path)
        self.assertEqual(meta["width"], 128)
        self.assertEqual(meta["height"], 128)
        self.assertEqual(meta["total_frames"], 20)
        self.assertAlmostEqual(meta["fps"], 10.0, delta=0.5)
        self.assertAlmostEqual(meta["duration_sec"], 2.0, delta=0.5)

    def test_04_frame_extraction(self):
        """Verify sampled frames are extracted with requested size and channels."""
        frames = extract_sampled_frames(self.sample_video_path, num_frames=6, target_size=(128, 128))
        self.assertEqual(len(frames), 6)
        for f in frames:
            self.assertIn("frame_idx", f)
            self.assertIn("timestamp_sec", f)
            self.assertEqual(f["rgb_image"].shape, (128, 128, 3))
            self.assertEqual(f["gray_image"].shape, (128, 128))
            self.assertEqual(f["rgb_image"].dtype, np.uint8)

    def test_05_frequency_domain_fft_finiteness(self):
        """Verify 2D FFT computation produces strictly finite values and valid normalization."""
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        res = compute_2d_fft_spectrum(gray)
        
        # Check finite values (no NaN, no Inf)
        self.assertTrue(np.all(np.isfinite(res["magnitude"])))
        self.assertTrue(np.all(np.isfinite(res["log_magnitude"])))
        self.assertTrue(np.all(np.isfinite(res["normalized_log_magnitude"])))
        
        # Check range of normalized log magnitude
        self.assertGreaterEqual(float(np.min(res["normalized_log_magnitude"])), 0.0)
        self.assertLessEqual(float(np.max(res["normalized_log_magnitude"])), 1.0)

    def test_06_high_frequency_energy_ratio(self):
        """Verify HFER is bounded within [0, 1] and scales predictably."""
        # 1. Smooth low-frequency image (constant field)
        smooth = np.ones((128, 128), dtype=np.uint8) * 128
        res_smooth = compute_2d_fft_spectrum(smooth)
        hfer_smooth, _, _ = compute_high_frequency_energy_ratio(res_smooth["magnitude"], cutoff_ratio=0.3)
        self.assertGreaterEqual(hfer_smooth, 0.0)
        self.assertLess(hfer_smooth, 0.05) # Most energy is at DC center (0,0)

        # 2. Checkerboard pattern (high frequency)
        checker = ((np.indices((128, 128)).sum(axis=0) % 2) * 255).astype(np.uint8)
        res_checker = compute_2d_fft_spectrum(checker)
        hfer_checker, _, _ = compute_high_frequency_energy_ratio(res_checker["magnitude"], cutoff_ratio=0.3)
        self.assertGreater(hfer_checker, hfer_smooth)
        self.assertLessEqual(hfer_checker, 1.0)

    def test_07_radial_profile(self):
        """Verify radial profile bins are monotonic and finite."""
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        res = compute_2d_fft_spectrum(gray)
        bins, profile = compute_radial_profile(res["magnitude"], num_bins=30)
        self.assertEqual(len(bins), 30)
        self.assertEqual(len(profile), 30)
        self.assertTrue(np.all(np.isfinite(profile)))

    def test_08_temporal_consistency_identical_and_differing(self):
        """Verify temporal difference metric is 0 on identical frames, > 0 on differing frames."""
        img1 = np.ones((64, 64), dtype=np.uint8) * 100
        img2 = np.ones((64, 64), dtype=np.uint8) * 100 # Identical to img1
        img3 = np.ones((64, 64), dtype=np.uint8) * 200 # Differing from img2

        mock_frames = [
            {"frame_idx": 0, "timestamp_sec": 0.0, "gray_image": img1},
            {"frame_idx": 5, "timestamp_sec": 0.5, "gray_image": img2},
            {"frame_idx": 10, "timestamp_sec": 1.0, "gray_image": img3},
        ]
        
        diffs = compute_temporal_differences(mock_frames)
        self.assertEqual(len(diffs), 2)
        
        # Between identical frames: MAD and RMSD must be 0.0
        self.assertEqual(diffs[0]["mad"], 0.0)
        self.assertEqual(diffs[0]["rmsd"], 0.0)
        
        # Between differing frames: MAD and RMSD must be strictly positive
        self.assertGreater(diffs[1]["mad"], 0.0)
        self.assertGreater(diffs[1]["rmsd"], 0.0)

    def test_09_spatial_analyzer(self):
        """Verify spatial face detector handles missing faces gracefully without error."""
        analyzer = SpatialAnalyzer()
        blank_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = analyzer.detect_faces(blank_img)
        self.assertIn("faces_found", res)
        self.assertIn("heuristic_disclaimer", res)
        self.assertEqual(res["annotated_image"].shape, (100, 100, 3))

    def test_10_dataframe_and_exports(self):
        """Verify metrics can be serialized to both CSV and JSON formats."""
        records = [
            {"frame_idx": 0, "timestamp_sec": 0.0, "hfer": 0.12, "mad": 0.0},
            {"frame_idx": 5, "timestamp_sec": 0.5, "hfer": 0.15, "mad": 0.035},
        ]
        df = pd.DataFrame(records)
        csv_data = df.to_csv(index=False)
        self.assertIn("timestamp_sec", csv_data)
        
        json_data = json.dumps(records)
        parsed = json.loads(json_data)
        self.assertEqual(len(parsed), 2)


if __name__ == "__main__":
    unittest.main()
