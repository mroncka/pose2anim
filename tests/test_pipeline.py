"""Tests for pose2anim pipeline."""

import numpy as np
import pytest


def test_yolo_estimator_init():
    """Test YOLO estimator can be initialized."""
    from pose2anim.pose2d.yolo_estimator import YOLOPoseEstimator

    config = {"model": "yolo26n-pose", "confidence": 0.5, "device": "cpu"}
    estimator = YOLOPoseEstimator(config)
    assert estimator.model is not None


def test_d3dp_lifter_init():
    """Test D3DP lifter can be initialized."""
    from pose2anim.pose3d.d3dp_lifter import D3DPLifter

    config = {"num_proposals": 3, "sampling_timesteps": 2}
    lifter = D3DPLifter(config)
    assert lifter.model is not None


def test_d3dp_lift_shape():
    """Test D3DP output shape matches input."""
    from pose2anim.pose3d.d3dp_lifter import D3DPLifter

    config = {"num_proposals": 3, "sampling_timesteps": 2}
    lifter = D3DPLifter(config)

    # Fake 2D keypoints: 10 frames, 17 joints, (x, y, conf)
    kp2d = np.random.rand(10, 17, 3).astype(np.float32)
    kp3d = lifter.lift(kp2d)

    assert kp3d.shape == (10, 17, 3)


def test_bvh_writer():
    """Test BVH writer produces valid file."""
    import tempfile
    from pathlib import Path
    from pose2anim.export.bvh_writer import BVHWriter

    config = {"fps": 30, "smooth": False}
    writer = BVHWriter(config)

    # Fake 3D keypoints
    kp3d = np.random.rand(5, 17, 3).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = str(Path(tmpdir) / "test.bvh")
        result = writer.write(kp3d, output)

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "HIERARCHY" in content
        assert "MOTION" in content
        assert "Frames: 5" in content


def test_skeleton_definitions():
    """Test skeleton definitions are consistent."""
    from pose2anim.utils.skeleton import (
        COCO17_JOINT_NAMES, COCO17_SKELETON, H36M_JOINT_NAMES, COCO_TO_H36M,
    )

    assert len(COCO17_JOINT_NAMES) == 17
    assert len(H36M_JOINT_NAMES) == 17

    for parent, child in COCO17_SKELETON:
        assert 0 <= parent < 17
        assert 0 <= child < 17

    for coco_idx, h36m_idx in COCO_TO_H36M.items():
        assert 0 <= coco_idx < 17
        assert 0 <= h36m_idx < 17
