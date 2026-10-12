"""
Spatial Analyzer Module for Deepfake Forensic Prototype.
Performs face localization and spatial appearance inspection using OpenCV cascades.
Includes strict disclaimers regarding heuristic limitations.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
# pyrefly: ignore [missing-import]
import cv2
import numpy as np


class SpatialAnalyzer:
    """
    Analyzes spatial appearance of video frames with optional face detection.
    """
    def __init__(self):
        # 1. Check local directory first
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
        cv_path = getattr(cv2.data, 'haarcascades', '') + "haarcascade_frontalface_default.xml"
        
        target_path = None
        if os.path.exists(local_path):
            target_path = local_path
        elif os.path.exists(cv_path):
            target_path = cv_path
        else:
            # Fallback: auto-download from official OpenCV repository
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, local_path)
                if os.path.exists(local_path):
                    target_path = local_path
            except Exception:
                target_path = None

        if target_path and os.path.exists(target_path):
            self.face_cascade = cv2.CascadeClassifier(target_path)
            self.detector_available = not self.face_cascade.empty()
        else:
            self.face_cascade = None
            self.detector_available = False

    def detect_faces(
        self,
        rgb_image: np.ndarray,
        scale_factor: float = 1.08,
        min_neighbors: int = 3,
        min_size: Tuple[int, int] = (20, 20)
    ) -> Dict[str, Any]:
        """
        Locates faces in an RGB image using the Haar Cascade detector.
        
        Returns:
            Dict containing:
                - 'faces_found': int (count of detected faces)
                - 'annotated_image': np.ndarray (RGB image with drawn bounding boxes)
                - 'bounding_boxes': List of [x, y, w, h]
                - 'face_crops': List of np.ndarray
                - 'status_message': Descriptive note for interface display
                - 'heuristic_disclaimer': Scientific limitation disclaimer
        """
        if not self.detector_available or self.face_cascade is None:
            return {
                "faces_found": 0,
                "annotated_image": rgb_image.copy(),
                "bounding_boxes": [],
                "face_crops": [],
                "status_message": "OpenCV Haar cascade classifier data file is unavailable.",
                "heuristic_disclaimer": (
                    "Face detection failure is NOT evidence of manipulation. "
                    "Haar cascades depend on frontal illumination, pose, and resolution."
                )
            }

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        annotated_image = rgb_image.copy()
        face_crops = []
        bounding_boxes = []

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                bounding_boxes.append((int(x), int(y), int(w), int(h)))
                # Draw bounding box in vibrant cyan for modern UI styling
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 220, 255), 2)
                cv2.putText(
                    annotated_image,
                    "Face ROI",
                    (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA
                )
                crop = rgb_image[y:y+h, x:x+w]
                if crop.size > 0:
                    face_crops.append(crop)

            status_msg = f"Detected {len(faces)} face region(s) in frame."
        else:
            status_msg = "No frontal face detected by Haar cascade."

        return {
            "faces_found": len(faces),
            "annotated_image": annotated_image,
            "bounding_boxes": bounding_boxes,
            "face_crops": face_crops,
            "status_message": status_msg,
            "heuristic_disclaimer": (
                "Notice: Face detector outcomes (e.g. non-detection, partial bounding box) "
                "stem from viewing angle, low contrast, or motion blur. They do NOT provide "
                "evidence that the video has been manipulated."
            )
        }
