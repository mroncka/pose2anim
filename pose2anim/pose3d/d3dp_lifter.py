"""D3DP-based 3D pose lifting from 2D keypoints."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class D3DPLifter:
    """Diffusion-based 3D Human Pose Estimation (D3DP, ICCV 2023).

    Lifts 2D pose sequences to 3D using a conditional diffusion model
    with Joint-wise reProjection-based Multi-hypothesis Aggregation (JPMA).

    Args:
        config: Configuration dict with keys:
            - num_proposals: Number of 3D hypotheses per frame (default: 5)
            - sampling_timesteps: Diffusion denoising steps (default: 5)
            - model_checkpoint: Path to pretrained weights
            - dataset: Training dataset (h36m or mpi_inf_3dhp)
    """

    def __init__(self, config: dict):
        self.config = config
        self.num_proposals = config.get("num_proposals", 5)
        self.sampling_timesteps = config.get("sampling_timesteps", 5)
        self.model = None
        self.device = self._resolve_device()
        self._load_model()

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load D3DP model and pretrained weights."""
        checkpoint_path = self.config.get("model_checkpoint")

        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading D3DP checkpoint: {checkpoint_path}")
            self.model = self._build_model()
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                "No D3DP checkpoint found. "
                "Download pretrained weights from: "
                "https://github.com/paTRICK-swk/D3DP#pretrained-models"
            )
            self.model = self._build_model()

        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        """Build the D3DP diffusion model architecture.

        TODO: Integrate actual D3DP model from the paper's codebase.
        This is a placeholder that will be replaced with the full
        GaussianDiffusion + ST-GCN architecture.
        """
        # Placeholder - actual implementation requires:
        # 1. Spatio-Temporal GCN backbone
        # 2. Gaussian Diffusion process
        # 3. JPMA aggregation module
        logger.info("Building D3DP model architecture...")

        from pose2anim.pose3d._model_stub import D3DPModelStub
        return D3DPModelStub(
            num_joints=17,
            num_proposals=self.num_proposals,
            sampling_timesteps=self.sampling_timesteps,
        )

    def lift(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """Lift 2D keypoints to 3D.

        Args:
            keypoints_2d: 2D keypoints (T, 17, 3) with [x, y, conf].

        Returns:
            3D keypoints (T, 17, 3) with [x, y, z].
        """
        # Normalize 2D inputs
        xy = keypoints_2d[:, :, :2].copy()
        confidences = keypoints_2d[:, :, 2:]

        # Center and scale normalization
        xy = self._normalize_2d(xy)

        # Convert to tensor
        xy_tensor = torch.from_numpy(xy).float().to(self.device)

        with torch.no_grad():
            # Generate multiple 3D hypotheses via diffusion
            hypotheses = self.model.sample(
                xy_tensor,
                num_proposals=self.num_proposals,
                sampling_timesteps=self.sampling_timesteps,
            )
            # hypotheses shape: (num_proposals, T, 17, 3)

            # JPMA: Select best joint per hypothesis via 2D reprojection
            keypoints_3d = self._jpma_aggregate(hypotheses, xy_tensor)

        return keypoints_3d.cpu().numpy()  # (T, 17, 3)

    def _normalize_2d(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize 2D keypoints to [-1, 1] range."""
        # Use hip center (joint 0 in H36M) as reference
        center = keypoints.mean(axis=1, keepdims=True)
        keypoints = keypoints - center
        scale = np.abs(keypoints).max()
        if scale > 0:
            keypoints = keypoints / scale
        return keypoints

    def _jpma_aggregate(
        self,
        hypotheses: torch.Tensor,
        gt_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Joint-wise reProjection-based Multi-hypothesis Aggregation.

        For each joint, select the hypothesis whose 2D reprojection
        is closest to the original 2D detection.

        Args:
            hypotheses: (num_proposals, T, 17, 3) 3D pose hypotheses.
            gt_2d: (T, 17, 2) original 2D keypoints.

        Returns:
            Aggregated 3D pose (T, 17, 3).
        """
        num_proposals, T, num_joints, _ = hypotheses.shape

        # Simple orthographic projection for reprojection
        projected = hypotheses[:, :, :, :2]  # (P, T, 17, 2)

        # Compute per-joint reprojection error
        gt_2d_expanded = gt_2d.unsqueeze(0).expand_as(projected)
        errors = ((projected - gt_2d_expanded) ** 2).sum(dim=-1)  # (P, T, 17)

        # Select best hypothesis per joint
        best_indices = errors.argmin(dim=0)  # (T, 17)

        # Gather best joints
        result = torch.zeros(T, num_joints, 3, device=hypotheses.device)
        for t in range(T):
            for j in range(num_joints):
                result[t, j] = hypotheses[best_indices[t, j], t, j]

        return result
