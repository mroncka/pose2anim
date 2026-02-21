"""BVH animation file writer."""

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose2anim.utils.skeleton import COCO17_SKELETON, COCO17_JOINT_NAMES, BVH_HIERARCHY

logger = logging.getLogger(__name__)


class BVHWriter:
    """Export 3D keypoints to BVH motion capture format.

    Converts 3D joint positions to joint rotations and writes
    standard BVH files compatible with Blender, Maya, MotionBuilder.
    """

    def __init__(self, config: dict):
        self.config = config
        self.fps = config.get("fps", 30)
        self.smooth = config.get("smooth", True)
        self.smooth_window = config.get("smooth_window", 5)

    def write(self, keypoints_3d: np.ndarray, output_path: str) -> str:
        """Write 3D keypoints to BVH file.

        Args:
            keypoints_3d: 3D joint positions (T, 17, 3).
            output_path: Output file path.

        Returns:
            Path to written BVH file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.smooth:
            keypoints_3d = self._temporal_smooth(keypoints_3d)

        # Convert positions to rotations
        rotations = self._positions_to_rotations(keypoints_3d)

        # Write BVH
        frame_time = 1.0 / self.fps
        num_frames = keypoints_3d.shape[0]

        with open(output_path, "w") as f:
            # Write hierarchy
            f.write(BVH_HIERARCHY)

            # Write motion data
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            for frame_idx in range(num_frames):
                # Root position + all joint rotations (Euler ZXY)
                root_pos = keypoints_3d[frame_idx, 0]  # Hip center
                values = [f"{root_pos[0]:.6f}", f"{root_pos[1]:.6f}", f"{root_pos[2]:.6f}"]

                for joint_rot in rotations[frame_idx]:
                    euler = joint_rot  # Already in degrees
                    values.extend([f"{euler[0]:.6f}", f"{euler[1]:.6f}", f"{euler[2]:.6f}"])

                f.write(" ".join(values) + "\n")

        logger.info(f"Written BVH: {output_path} ({num_frames} frames @ {self.fps}fps)")
        return str(output_path)

    def _positions_to_rotations(self, positions: np.ndarray) -> np.ndarray:
        """Convert 3D positions to joint rotations.

        Uses inverse kinematics to compute Euler angles from
        joint position differences along the skeleton hierarchy.

        Args:
            positions: (T, 17, 3) joint positions.

        Returns:
            (T, 17, 3) Euler rotations in degrees (ZXY order).
        """
        T, num_joints, _ = positions.shape
        rotations = np.zeros((T, num_joints, 3))

        for t in range(T):
            for parent, child in COCO17_SKELETON:
                bone_vec = positions[t, child] - positions[t, parent]
                bone_len = np.linalg.norm(bone_vec)

                if bone_len > 1e-6:
                    bone_dir = bone_vec / bone_len
                    # Compute rotation from reference direction to bone direction
                    ref_dir = np.array([0, 1, 0])  # Y-up reference
                    rot = self._direction_to_euler(ref_dir, bone_dir)
                    rotations[t, child] = rot

        return rotations

    @staticmethod
    def _direction_to_euler(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute Euler angles (ZXY) to rotate ref direction to target."""
        cross = np.cross(ref, target)
        dot = np.dot(ref, target)

        if np.linalg.norm(cross) < 1e-6:
            return np.zeros(3)

        angle = np.arccos(np.clip(dot, -1, 1))
        axis = cross / np.linalg.norm(cross)

        rot = Rotation.from_rotvec(axis * angle)
        euler = rot.as_euler("ZXY", degrees=True)
        return euler

    def _temporal_smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce jitter."""
        from scipy.ndimage import uniform_filter1d

        window = self.smooth_window
        smoothed = uniform_filter1d(keypoints, size=window, axis=0, mode="nearest")
        return smoothed
