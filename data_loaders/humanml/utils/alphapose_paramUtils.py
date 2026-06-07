import numpy as np


ALPHA_POSE_JOINT_NAMES = [
    'pelvis',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'neck',
    'nose',
    'head',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
]

alpha_pose_raw_offsets = np.array([
    [0,0,0], # 0-root
    [1,0,0], # 1-RHip
    [0,-1,0], # 2-RKnee
    [0,-1,0], # 3-RAnkle
    [-1,0,0], # 4-LHip
    [0,-1,0], # 5-LKnee
    [0,-1,0], # 6-LAnkle
    [0,1,0], # 7-neck
    [0,1,0], # 8-nose
    [0,1,0], # 9-head
    [-1,0,0], # 10-LShoulder
    [-1,0,0], # 11-LElbow
    [-1,0,0], # 12-LWrist
    [1,0,0], # 13-RShoulder
    [1,0,0], # 14-RElbow
    [1,0,0], # 15-RWrist
])

alpha_pose_kinematic_chain = [
    [0, 1, 2, 3], # Right Leg
    [0, 4, 5, 6], # Left Leg
    [0, 7, 8, 9], # Body
    [7, 13, 14, 15], # Right Arm
    [7, 10, 11, 12] # Left Arm
]
