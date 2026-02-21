"""Debug visualization utilities."""

import numpy as np


def draw_3d_skeleton(keypoints_3d: np.ndarray, skeleton: list, ax=None):
    """Plot 3D skeleton using matplotlib.

    Args:
        keypoints_3d: (17, 3) joint positions.
        skeleton: List of (parent, child) bone connections.
        ax: Optional matplotlib 3D axis.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        return

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Plot joints
    ax.scatter(
        keypoints_3d[:, 0],
        keypoints_3d[:, 1],
        keypoints_3d[:, 2],
        c="red", s=20,
    )

    # Plot bones
    for parent, child in skeleton:
        ax.plot(
            [keypoints_3d[parent, 0], keypoints_3d[child, 0]],
            [keypoints_3d[parent, 1], keypoints_3d[child, 1]],
            [keypoints_3d[parent, 2], keypoints_3d[child, 2]],
            c="blue", linewidth=2,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax
