"""Stub D3DP model for development/testing.

Replace with actual D3DP implementation from:
https://github.com/paTRICK-swk/D3DP
"""

import torch
import torch.nn as nn


class D3DPModelStub(nn.Module):
    """Placeholder D3DP model for pipeline testing.

    This generates random 3D poses for pipeline validation.
    Replace with the actual GaussianDiffusion + ST-GCN model
    from the D3DP paper for production use.
    """

    def __init__(self, num_joints: int = 17, num_proposals: int = 5,
                 sampling_timesteps: int = 5):
        super().__init__()
        self.num_joints = num_joints
        self.num_proposals = num_proposals
        self.sampling_timesteps = sampling_timesteps

        # Simple linear baseline for testing
        self.lifter = nn.Sequential(
            nn.Linear(num_joints * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3),
        )

    def sample(self, keypoints_2d: torch.Tensor, num_proposals: int = 5,
               sampling_timesteps: int = 5) -> torch.Tensor:
        """Generate 3D pose hypotheses.

        Args:
            keypoints_2d: (T, 17, 2) normalized 2D keypoints.
            num_proposals: Number of hypotheses to generate.
            sampling_timesteps: Diffusion steps (unused in stub).

        Returns:
            (num_proposals, T, 17, 3) 3D pose hypotheses.
        """
        T = keypoints_2d.shape[0]
        flat = keypoints_2d.reshape(T, -1)  # (T, 34)

        hypotheses = []
        for _ in range(num_proposals):
            # Add noise for diversity
            noisy = flat + torch.randn_like(flat) * 0.01
            pred = self.lifter(noisy)  # (T, 51)
            pred = pred.reshape(T, self.num_joints, 3)
            hypotheses.append(pred)

        return torch.stack(hypotheses)  # (P, T, 17, 3)

    def forward(self, x):
        return self.sample(x)
