"""YOLO26-based 2D pose estimation."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YOLOPoseEstimator:
    """2D human pose estimation using Ultralytics YOLO26-pose.

    Features:
        - NMS-free inference for lower latency
        - 17-point COCO keypoint detection
        - Small-Target-Aware Label Assignment (STAL)
        - Batch processing support
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO26 pose model."""
        try:
            from ultralytics import YOLO

            model_name = self.config.get("model", "yolo26m-pose")
            device = self.config.get("device", "auto")

            # Resolve device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading {model_name} on {device}")
            self.model = YOLO(f"{model_name}.pt")
            self.device = device

        except ImportError:
            logger.error("Ultralytics not installed. Run: pip install ultralytics")
            raise

    def estimate_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate 2D pose keypoints for a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Keypoints array (17, 3) with [x, y, confidence] per joint,
            or None if no person detected.
        """
        results = self.model(
            frame,
            device=self.device,
            conf=self.config.get("confidence", 0.5),
            verbose=False,
        )

        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.data
            if kps.shape[0] > 0:
                # Take highest-confidence person
                return kps[0].cpu().numpy()  # (17, 3)

        return None

    def estimate_video(self, video_path: str) -> np.ndarray:
        """Estimate 2D poses for all frames in a video.

        Args:
            video_path: Path to video file.

        Returns:
            Keypoints array (T, 17, 3) where T is number of frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {total_frames} frames from {video_path}")

        keypoints_all = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            kps = self.estimate_frame(frame)
            if kps is not None:
                keypoints_all.append(kps)
            else:
                # Interpolate or use zeros for missing frames
                if keypoints_all:
                    keypoints_all.append(keypoints_all[-1].copy())
                else:
                    keypoints_all.append(np.zeros((17, 3)))

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"  Frame {frame_idx}/{total_frames}")

        cap.release()

        return np.stack(keypoints_all)  # (T, 17, 3)
