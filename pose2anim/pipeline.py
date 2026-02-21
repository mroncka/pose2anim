"""Main pipeline orchestrator for pose2anim."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from pose2anim.pose2d.yolo_estimator import YOLOPoseEstimator
from pose2anim.pose3d.d3dp_lifter import D3DPLifter
from pose2anim.export.bvh_writer import BVHWriter
from pose2anim.utils.skeleton import COCO17_SKELETON

logger = logging.getLogger(__name__)


class Pose2AnimPipeline:
    """End-to-end video → 3D animation pipeline.

    Stages:
        1. YOLO26 2D pose estimation (per-frame keypoints)
        2. D3DP diffusion-based 3D lifting (2D → 3D)
        3. BVH/FBX animation export
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.pose2d = YOLOPoseEstimator(self.config["pose2d"])
        self.pose3d = D3DPLifter(self.config["pose3d"])
        self.exporter = BVHWriter(self.config["export"])

    def process_video(self, video_path: str, output_path: str) -> str:
        """Process a video file through the full pipeline.

        Args:
            video_path: Path to input video file.
            output_path: Path for output animation file.

        Returns:
            Path to the exported animation file.
        """
        logger.info(f"Processing video: {video_path}")

        # Stage 1: Extract 2D poses
        keypoints_2d = self.pose2d.estimate_video(video_path)
        logger.info(f"Extracted 2D poses: {keypoints_2d.shape[0]} frames")

        # Stage 2: Lift to 3D
        keypoints_3d = self.pose3d.lift(keypoints_2d)
        logger.info(f"Lifted to 3D: {keypoints_3d.shape}")

        # Stage 3: Export animation
        output_file = self.exporter.write(keypoints_3d, output_path)
        logger.info(f"Exported animation: {output_file}")

        return output_file

    def process_live(self, camera_id: int = 0):
        """Process live webcam feed with real-time visualization.

        Args:
            camera_id: Camera device index.
        """
        cap = cv2.VideoCapture(camera_id)
        logger.info(f"Starting live capture from camera {camera_id}")

        frames_buffer = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 2D estimation (real-time)
                kp2d = self.pose2d.estimate_frame(frame)

                if kp2d is not None:
                    frames_buffer.append(kp2d)

                    # Lift when we have enough frames
                    if len(frames_buffer) >= self.config["pose3d"].get("window_size", 243):
                        batch = np.stack(frames_buffer[-243:])
                        kp3d = self.pose3d.lift(batch)
                        # TODO: real-time visualization of 3D skeleton
                        frames_buffer = frames_buffer[-200:]

                # Display 2D overlay
                if kp2d is not None:
                    self._draw_skeleton(frame, kp2d)

                cv2.imshow("pose2anim - live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw 2D skeleton overlay on frame."""
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.config["pose2d"]["confidence"]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        for j1, j2 in COCO17_SKELETON:
            if (keypoints[j1, 2] > self.config["pose2d"]["confidence"]
                    and keypoints[j2, 2] > self.config["pose2d"]["confidence"]):
                pt1 = tuple(keypoints[j1, :2].astype(int))
                pt2 = tuple(keypoints[j2, :2].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    @staticmethod
    def _default_config() -> dict:
        return {
            "pose2d": {
                "model": "yolo26m-pose",
                "confidence": 0.5,
                "device": "auto",
                "batch_size": 1,
            },
            "pose3d": {
                "method": "d3dp",
                "num_proposals": 5,
                "sampling_timesteps": 5,
                "model_checkpoint": None,
                "dataset": "h36m",
            },
            "export": {
                "format": "bvh",
                "fps": 30,
                "skeleton": "coco17",
                "smooth": True,
                "smooth_window": 5,
            },
        }
