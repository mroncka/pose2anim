"""Skeleton definitions and joint mappings."""

# COCO 17-keypoint skeleton
COCO17_JOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Bone connections (parent, child)
COCO17_SKELETON = [
    (0, 1), (0, 2),      # Nose → eyes
    (1, 3), (2, 4),      # Eyes → ears
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (5, 6),              # Shoulders
    (11, 12),            # Hips
    (5, 11), (6, 12),    # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Human3.6M 17-joint mapping (used by D3DP)
H36M_JOINT_NAMES = [
    "hip",            # 0
    "right_hip",      # 1
    "right_knee",     # 2
    "right_ankle",    # 3
    "left_hip",       # 4
    "left_knee",      # 5
    "left_ankle",     # 6
    "spine",          # 7
    "thorax",         # 8
    "neck",           # 9
    "head",           # 10
    "left_shoulder",  # 11
    "left_elbow",     # 12
    "left_wrist",     # 13
    "right_shoulder", # 14
    "right_elbow",    # 15
    "right_wrist",    # 16
]

# Mapping: COCO17 index → H36M index (approximate)
COCO_TO_H36M = {
    0: 10,   # nose → head
    5: 11,   # left_shoulder
    6: 14,   # right_shoulder
    7: 12,   # left_elbow
    8: 15,   # right_elbow
    9: 13,   # left_wrist
    10: 16,  # right_wrist
    11: 4,   # left_hip
    12: 1,   # right_hip
    13: 5,   # left_knee
    14: 2,   # right_knee
    15: 6,   # left_ankle
    16: 3,   # right_ankle
}

# BVH hierarchy definition for COCO17 skeleton
BVH_HIERARCHY = """HIERARCHY
ROOT Hip
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftHip
  {
    OFFSET -0.1 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftKnee
    {
      OFFSET 0.0 -0.4 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftAnkle
      {
        OFFSET 0.0 -0.4 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 -0.1 0.0
        }
      }
    }
  }
  JOINT RightHip
  {
    OFFSET 0.1 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightKnee
    {
      OFFSET 0.0 -0.4 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightAnkle
      {
        OFFSET 0.0 -0.4 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 -0.1 0.0
        }
      }
    }
  }
  JOINT Spine
  {
    OFFSET 0.0 0.2 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Neck
    {
      OFFSET 0.0 0.3 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Head
      {
        OFFSET 0.0 0.15 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 0.1 0.0
        }
      }
    }
    JOINT LeftShoulder
    {
      OFFSET -0.2 0.25 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftElbow
      {
        OFFSET -0.25 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftWrist
        {
          OFFSET -0.25 0.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET -0.1 0.0 0.0
          }
        }
      }
    }
    JOINT RightShoulder
    {
      OFFSET 0.2 0.25 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightElbow
      {
        OFFSET 0.25 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightWrist
        {
          OFFSET 0.25 0.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET 0.1 0.0 0.0
          }
        }
      }
    }
  }
}
"""
