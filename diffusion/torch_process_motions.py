import torch
from typing import Sequence, Tuple

from data_loaders.humanml.common.quaternion import qmul, qinv, qrot, quaternion_to_cont6d, qbetween
import data_loaders.humanml.scripts.motion_process as motion_process_module
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.scripts.motion_process import recover_from_ric

import numpy as np

n_joints = 22

global_torch_process_motion_feet_thre = 0.002
global_torch_process_motion_fid_r, global_torch_process_motion_fid_l   = [8, 11], [7, 10]
global_torch_process_motion_face_joint_indx   = [2, 1, 17, 16]
global_torch_process_motion_n_raw_offsets     = torch.from_numpy(paramUtil.t2m_raw_offsets)
global_torch_process_motion_kinematic_chain   = paramUtil.t2m_kinematic_chain

# define tgt_offsets (must be a torch.Tensor of shape [joints, 3])
example_path = "./assets/humanml_standard_motion_for_normalization.npy"
example_xyz = torch.from_numpy(np.load(example_path))  # shape [T, J, 3]
tgt_skel = Skeleton(global_torch_process_motion_n_raw_offsets, global_torch_process_motion_kinematic_chain, 'cpu')
global_torch_process_motion_tgt_offsets = tgt_skel.get_offsets_joints(example_xyz[0])


def get_skeleton_kwargs_from_dataset(dataset):
    if dataset == "nba":
        from data_loaders.humanml.utils.alphapose_paramUtils import alpha_pose_raw_offsets, alpha_pose_kinematic_chain, ALPHA_POSE_JOINT_NAMES
        kwargs = {
            'fid_r': [ALPHA_POSE_JOINT_NAMES.index('right_ankle'), ALPHA_POSE_JOINT_NAMES.index('right_ankle')], # Duplicate to match SMPL with 2 feet joints
            'fid_l': [ALPHA_POSE_JOINT_NAMES.index('left_ankle'), ALPHA_POSE_JOINT_NAMES.index('left_ankle')], # Duplicate to match SMPL with 2 feet joints
            'face_joint_idx': [
                ALPHA_POSE_JOINT_NAMES.index('right_hip'),
                ALPHA_POSE_JOINT_NAMES.index('left_hip'),
                ALPHA_POSE_JOINT_NAMES.index('right_shoulder'),
                ALPHA_POSE_JOINT_NAMES.index('left_shoulder')
            ],
            'raw_offsets': torch.from_numpy(alpha_pose_raw_offsets),
            'kinematic_chain': alpha_pose_kinematic_chain
        }
    elif dataset in ["video_mdm_synthetic", "video_mdm_synthetic_mvlift", "humanml", "egoexo", "fit3d", "egoexo,humanml", "fit3d,humanml", "fit3d_mvlift"]:
        kwargs = {
            'fid_r': global_torch_process_motion_fid_r,
            'fid_l': global_torch_process_motion_fid_l,
            'face_joint_idx': global_torch_process_motion_face_joint_indx,
            'raw_offsets': global_torch_process_motion_n_raw_offsets,
            'kinematic_chain': global_torch_process_motion_kinematic_chain
        }
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return kwargs

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

@torch.no_grad()
def _inverse_kinematics_torch(
    joints: torch.Tensor,                # (B, T, J, 3)
    raw_offsets: torch.Tensor,           # (J, 3) unit bone directions (like t2m_raw_offsets / kit_raw_offsets)
    kinematic_chain: Sequence[Sequence[int]],
    face_joint_idx: Sequence[int],       # [r_hip, l_hip, sdr_r, sdr_l]
) -> torch.Tensor:
    """
    Torch port of the IK routine used in motion_process.inverse_kinematics_np.
    Returns quaternions per joint: (B, T, J, 4), real part first.
    """
    B, T, J, _ = joints.shape
    device = joints.device
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx

    # Forward direction per frame
    across = (joints[..., r_hip, :] - joints[..., l_hip, :]) + (joints[..., sdr_r, :] - joints[..., sdr_l, :])
    across = _normalize(across)                                 # (B, T, 3)
    up = torch.tensor([0.0, 1.0, 0.0], device=device).view(1,1,3).expand(B, T, -1)
    forward = _normalize(torch.cross(up, across, dim=-1))       # (B, T, 3)

    # Root quaternion: rotate forward → +Z
    target = torch.tensor([0.0, 0.0, 1.0], device=device).view(1,1,3).expand(B, T, -1)
    root_quat = qbetween(forward, target)                       # (B, T, 4)

    # Allocate result
    quat_params = torch.zeros(B, T, J, 4, device=device, dtype=joints.dtype)
    quat_params[..., 0, :] = root_quat

    # Flatten batch+time for vectorized ops
    BT = B * T
    root_flat = root_quat.reshape(BT, 4)

    raw_offsets = raw_offsets.to(device).to(joints.dtype)       # (J, 3)
    u_all = raw_offsets                                         # unit rest directions

    # For each chain, walk down the hierarchy
    for chain in kinematic_chain:
        R = root_flat.clone()                                   # (BT, 4), reset at chain start
        for i in range(len(chain) - 1):
            p = chain[i]
            c = chain[i + 1]
            # u = rest-pose bone direction for bone p->c
            u = u_all[c].view(1, 3).expand(BT, -1)              # (BT, 3)
            # v = data bone direction for bone p->c
            Jp = joints[:, :, p, :].reshape(BT, 3)
            Jc = joints[:, :, c, :].reshape(BT, 3)
            v = _normalize(Jc - Jp)                             # (BT, 3)
            rot_u_v = qbetween(u, v)                            # (BT, 4)

            # Local rotation relative to parent orientation
            R_loc = qmul(qinv(R), rot_u_v)                      # (BT, 4)

            # Write out and advance orientation
            quat_params[:, :, c, :] = R_loc.view(B, T, 4)
            R = qmul(R, R_loc)

    return quat_params


def process_file_torch(
    positions: torch.Tensor,            # (B, T, J, 3) or (T, J, 3) – already canonical "global_positions"
    feet_thre: float = global_torch_process_motion_feet_thre,
    raw_offsets: torch.Tensor = global_torch_process_motion_n_raw_offsets,          # (J, 3) like t2m_raw_offsets / kit_raw_offsets (unit directions)
    kinematic_chain: Sequence[Sequence[int]] = global_torch_process_motion_kinematic_chain,
    face_joint_idx: Sequence[int] = global_torch_process_motion_face_joint_indx,      # [r_hip, l_hip, sdr_r, sdr_l]
    fid_r: Sequence[int] = global_torch_process_motion_fid_r,               # e.g. [8, 11] or dataset-specific
    fid_l: Sequence[int] = global_torch_process_motion_fid_l,               # e.g. [7, 10]
    return_everything: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch, batched reimplementation of motion_process.process_file starting from
    'New ground truth positions'.

    Returns:
      data           : (B, T-1, D) HumanML/KIT vector
      global_pos     : (B, T, J, 3) (same as input, forwarded)
      local_pos      : (B, T, J, 3) rotation-invariant local positions (RIC)
      l_velocity_xz  : (B, T-1, 2) root linear velocity (x,z) in root-local frame
    """
    if positions.dim() == 3:
        positions = positions.unsqueeze(0)                      # (1, T, J, 3)
    assert positions.dim() == 4 and positions.size(-1) == 3, f"positions must be (B, T, J, 3) but are {positions.shape}"

    B, T, J, _ = positions.shape
    device = positions.device
    global_positions = positions.clone()

    # ---------- Foot contacts (vel-based only, like original) ----------
    # left feet: two joints; right feet: two joints
    def _feet_contacts(pos: torch.Tensor, fids: Sequence[int]) -> torch.Tensor:
        # pos: (B, T, J, 3)
        d = pos[:, 1:, fids, :] - pos[:, :-1, fids, :]          # (B, T-1, 2, 3)
        vel2 = (d ** 2).sum(dim=-1)                             # (B, T-1, 2)
        th = torch.tensor([feet_thre, feet_thre], device=device, dtype=pos.dtype).view(1, 1, 2)
        return (vel2 < th).to(pos.dtype)                        # (B, T-1, 2)

    feet_l = _feet_contacts(global_positions, fid_l)            # (B, T-1, 2)
    feet_r = _feet_contacts(global_positions, fid_r)            # (B, T-1, 2)

    # ---------- IK → quaternions (per joint), then cont6d ----------
    quat_params = _inverse_kinematics_torch(
        global_positions, raw_offsets, kinematic_chain, face_joint_idx
    )                                                           # (B, T, J, 4)
    cont6d_params = quaternion_to_cont6d(quat_params)           # (B, T, J, 6)
    r_rot = quat_params[..., 0, :]                              # (B, T, 4)

    # ---------- Root linear & angular velocities ----------
    root_vel = global_positions[:, 1:, 0, :] - global_positions[:, :-1, 0, :]  # (B, T-1, 3)
    root_vel_local = qrot(r_rot[:, 1:, :], root_vel)            # (B, T-1, 3)
    l_velocity_xz = root_vel_local[..., [0, 2]]                 # (B, T-1, 2)

    r_velocity_quat = qmul(r_rot[:, 1:, :], qinv(r_rot[:, :-1, :]))  # (B, T-1, 4)
    # rotation velocity along y-axis (original used arcsin(q.z))
    # Clamp the value to the valid range [-1, 1] before passing to arcsin
    rot_vel_y = torch.arcsin(torch.clamp(r_velocity_quat[..., 2:3], -1.0, 1.0)) # (B, T-1, 1)
    #rot_vel_y = torch.arcsin(r_velocity_quat[..., 2:3])         # (B, T-1, 1) old version, introduces "nan"s

    # ---------- RIFKE local positions (subtract root xz, rotate to face +Z) ----------
    local_pos = global_positions.clone()
    local_pos[..., 0] -= global_positions[..., 0:1, 0]          # x
    local_pos[..., 2] -= global_positions[..., 0:1, 2]          # z
    r_rep = r_rot.unsqueeze(-2).expand(-1, -1, J, -1)           # (B, T, J, 4)
    local_pos = qrot(r_rep, local_pos)                          # (B, T, J, 3)

    # Root height
    root_y = local_pos[..., 0, 1:2]                             # (B, T, 1)

    # ---------- Pack features like original ----------
    # rot_data: cont6d for non-root joints
    rot_data = cont6d_params[..., 1:, :].reshape(B, T, -1)      # (B, T, (J-1)*6)
    # ric_data: local positions for non-root joints
    ric_data = local_pos[..., 1:, :].reshape(B, T, -1)          # (B, T, (J-1)*3)
    # local velocity for all joints in root-local frame
    delta_all = global_positions[:, 1:, :, :] - global_positions[:, :-1, :, :]     # (B, T-1, J, 3)
    r_rep_vel = r_rot[:, :-1, :].unsqueeze(-2).expand(-1, -1, J, -1)               # (B, T-1, J, 4)
    local_vel = qrot(r_rep_vel, delta_all).reshape(B, T-1, -1)                      # (B, T-1, J*3)

    # root_data = [rot_vel_y, l_velocity_xz, root_y[:-1]]
    root_data = torch.cat(
        [rot_vel_y, l_velocity_xz, root_y[:, :-1, :]], dim=-1
    )                                                                               # (B, T-1, 1+2+1)

    # Align to T-1 everywhere and concatenate in the same order as original
    feats = [
        root_data,
        ric_data[:, :-1, :],
        rot_data[:, :-1, :],
        local_vel,
        feet_l,           # (B, T-1, 2)
        feet_r            # (B, T-1, 2)
    ]
    # D = 4 + 3 * (J - 1) + 6 * (J - 1) + 3 * J + 2 + 2 = 12 * J - 1
    data = torch.cat(feats, dim=-1)                                                 # (B, T-1, D)

    if return_everything:
        return (
            data, global_positions, positions, l_velocity_xz, feet_l, feet_r, rot_vel_y, r_rot,
            cont6d_params, root_vel, ric_data, rot_data, root_data, local_vel, local_pos
        )
    else:
        return data, global_positions, local_pos, l_velocity_xz

def no_frame_loss_process_file_torch(xyz, **kwargs):
    #0   , 1               , 2        , 3            , 4     , 5     , 6        , 7    , 8            , 9       , 10      , 11      , 12       , 13       , 14
    #data, global_positions, positions, l_velocity_xz, feet_l, feet_r, rot_vel_y, r_rot, cont6d_params, root_vel, ric_data, rot_data, root_data, local_vel, local_pos
    kwargs['return_everything'] = True # Always return everything
    res_tup = process_file_torch(xyz, **kwargs)
    data = res_tup[0] # (B, T-1, D)
    ric_data = res_tup[10]
    rot_data = res_tup[11]
    local_pos = res_tup[14]
    B, T_minus_1, D = data.shape
    results = torch.zeros(B, T_minus_1 + 1, D, device=xyz.device, dtype=xyz.dtype) # Last frame velocites is 0, so initialize with 0s
    results[:, :-1, :] = data # Copy the data for all but last frame

    n_joints = xyz.shape[2]

    ric_idx = 4
    rot_idx = 4 + (n_joints-1)*3
    vel_idx = rot_idx + (n_joints-1)*6
    feet_idx = vel_idx + 3*n_joints

    results[:, -1, 3] = local_pos[:, -1, 0, 1]           # Add the last frame of root_y
    results[:, -1, ric_idx:rot_idx] = ric_data[:, -1, :] # Add the last frame of ric_data
    results[:, -1, rot_idx:vel_idx] = rot_data[:, -1, :] # Add the last frame of rot_data
    results[:, -1, feet_idx:] = data[:, -1, feet_idx:]   # The feet contact values are copied from the last frame of data (frame T-1) as they are velocity-based and we don't kown the "future" frames
    return results


def mask_refactor_to_hmlvec(mask,
    fid_r: Sequence[int] = global_torch_process_motion_fid_r,               # e.g. [8, 11] or dataset-specific
    fid_l: Sequence[int] = global_torch_process_motion_fid_l,               # e.g. [7, 10]
    keep_shape: bool = False,
    **kwargs
):
    """
    Refactors a joint-based mask (B, J, 1, T) into a HumanML/KIT vector mask (B, D, 1, T - 1).
    See process_file_torch for more details, as this is a matching implementation to broadcast the mask to the vector format.
    """
    B, J, _, T = mask.shape
    D_total = 12 * J - 1
    if not keep_shape:
        last_dim = T - 1
    else:
        last_dim = T
        # add 0s as frame T+1 for easy calculations
        mask = torch.cat([mask, torch.zeros(B, J, 1, 1, device=mask.device, dtype=mask.dtype)], dim=-1)
    mask_hmlvec = torch.zeros(B, D_total, 1, last_dim, device=mask.device, dtype=mask.dtype)
    D_idx = 0

    mask_hmlvec[:, D_idx : D_idx + 4, :, :] = mask[:, [0], :, :-1]
    D_idx += 4

    indexes = [i for i in range(1, J) for _ in range(3)]
    mask_hmlvec[:, D_idx : D_idx + (J-1) * 3, :, :] = mask[:, indexes, :, :-1]
    D_idx += (J-1) * 3

    indexes = [i for i in range(1, J) for _ in range(6)]
    mask_hmlvec[:, D_idx : D_idx + (J-1) * 6, :, :] = mask[:, indexes, :, :-1]
    D_idx += (J-1) * 6

    indexes = [i for i in range(0, J) for _ in range(3)]
    mask_hmlvec[:, D_idx : D_idx + J * 3, :, :] = mask[:, indexes, :, :-1] * mask[:, indexes, :, 1:]
    D_idx += J * 3

    mask_hmlvec[:, D_idx : D_idx + 2, :, :] = mask[:, fid_l, :, :-1] * mask[:, fid_l, :, 1:]
    D_idx += 2

    mask_hmlvec[:, D_idx : D_idx + 2, :, :] = mask[:, fid_r, :, :-1] * mask[:, fid_r, :, 1:]
    D_idx += 2
    
    return mask_hmlvec


def set_motion_process_module_globals():
    # match the config used in that script's HumanML block
    motion_process_module.l_idx1, motion_process_module.l_idx2 = 5, 8
    motion_process_module.fid_r, motion_process_module.fid_l   = [8, 11], [7, 10]
    motion_process_module.face_joint_indx   = [2, 1, 17, 16]
    motion_process_module.joints_num = n_joints
    motion_process_module.n_raw_offsets     = torch.from_numpy(paramUtil.t2m_raw_offsets)
    motion_process_module.kinematic_chain   = paramUtil.t2m_kinematic_chain

    # define tgt_offsets (must be a torch.Tensor of shape [joints, 3])
    example_path = "./assets/humanml_standard_motion_for_normalization.npy"
    example_xyz = torch.from_numpy(np.load(example_path))  # shape [T, J, 3]
    tgt_skel = Skeleton(motion_process_module.n_raw_offsets, motion_process_module.kinematic_chain, 'cpu')
    motion_process_module.tgt_offsets = tgt_skel.get_offsets_joints(example_xyz[0])


def nan_report(np_data, torch_data_np, J=22):
    # feature block sizes (must match packing): root(4), ric(63), rot(126), lvel(66), feet_l(2), feet_r(2)
    blocks = [
        ("root", slice(0, 4)),
        ("ric",  slice(4, 4 + (J-1)*3)),             # 63
        ("rot",  slice(4 + (J-1)*3, 4 + (J-1)*3 + (J-1)*6)),  # 126
        ("lvel", slice(4 + 9*(J-1), 4 + 9*(J-1) + 3*J)),      # 66
        ("feet_l", slice(4 + 9*(J-1) + 3*J, 4 + 9*(J-1) + 3*J + 2)),
        ("feet_r", slice(4 + 9*(J-1) + 3*J + 2, 4 + 9*(J-1) + 3*J + 4)),
    ]
    B, Tm1, D = np_data.shape
    out = []
    for b in range(B):
        row = {"batch": b}
        any_nan = False
        for name, sl in blocks:
            n1 = np.isnan(np_data[b, :, sl]).sum()
            n2 = np.isnan(torch_data_np[b, :, sl]).sum()
            row[name+"_np_nan"] = int(n1)
            row[name+"_th_nan"] = int(n2)
            any_nan = any_nan or n1>0 or n2>0
        row["any_nan"] = any_nan
        out.append(row)
    return out

def strip_trailing_zeros_btj3(x):  # x: (B,T,J,3)
    B,T,J,_ = x.shape
    out = []
    lengths = []
    for b in range(B):
        nonzero = (np.abs(x[b]).sum(axis=(1,2)) > 0)
        if not nonzero.any():  # all zeros, keep 1 frame to avoid empty
            L = 1
        else:
            L = np.where(nonzero)[0][-1] + 1
        out.append(x[b, :L])
        lengths.append(L)
    return out, lengths


names = [
    "data", "global_positions", "positions", "l_velocity", "feet_l", "feet_r", "r_velocity", "r_rot",
    "cont_6d_params", "velocity", "ric_data", "rot_data", "root_data", "local_vel"
]
def compare_np_torch_features(
    motion_np: np.ndarray,                 # (T,J,3) or (B,T,J,3) — already normalized!
    feet_thre: float,
    raw_offsets,                           # (J,3) np.ndarray or torch.Tensor (unit bone directions)
    kinematic_chain,                       # list[list[int]]
    face_joint_idx,                        # [r_hip, l_hip, sdr_r, sdr_l]
    fid_r,                                 # right-foot joint ids (e.g., [8, 11])
    fid_l,                                 # left-foot joint ids  (e.g., [7, 10])
    atol: float = 1e-4,
    rtol: float = 1e-3,
    return_arrays: bool = False,
):
    """
    Compares NumPy extract_features (starting at 'New ground truth positions')
    with the pure-Torch process_file_torch, and reports similarity metrics.

    Returns a dict with per-batch metrics (max_abs_diff, mean_abs_diff, allclose),
    and optionally the aligned arrays.
    """

    # ---- Normalize shapes to batched form ----
    if motion_np.ndim == 3:
        motion_np = motion_np[None, ...]  # (1,T,J,3)
    assert motion_np.ndim == 4 and motion_np.shape[-1] == 3, "motion_np must be (B,T,J,3) or (T,J,3)"

    B, T, J, _ = motion_np.shape

    # ---- NumPy: run extract_features per sequence (it is unbatched) ----
    np_datas = []
    for b in range(B):
        seq = motion_np[b]  # (T,J,3), already normalized
        np_feat_tup = motion_process_module.extract_features(
            seq.astype(np.float32, copy=True),
            feet_thre,
            torch.from_numpy(raw_offsets).float(),              # for NumPy path this can be np.ndarray
            kinematic_chain,
            face_joint_idx,
            fid_r,
            fid_l,
            smooth_forward=False,
            return_everything=True
        )
        np_feat = np_feat_tup[0]                           # (T-1, D)
        np_datas.append(np_feat.astype(np.float32, copy=False))
    np_data = np.stack(np_datas, axis=0)  # (B, T-1, D)

    # ---- Torch: run our batched implementation ----
    if isinstance(raw_offsets, np.ndarray):
        raw_offsets_t = torch.from_numpy(raw_offsets).float()
    else:
        raw_offsets_t = raw_offsets.float()

    positions_t = torch.from_numpy(motion_np).float()          # (B,T,J,3)
    print("positions_t", positions_t.shape)
    torch_data_tup = process_file_torch(
        positions=positions_t,
        feet_thre=feet_thre,
        raw_offsets=raw_offsets_t,
        kinematic_chain=kinematic_chain,
        face_joint_idx=face_joint_idx,
        fid_r=fid_r,
        fid_l=fid_l,
        return_everything=True
    )  # (B, T-1, D_torch)
    print("torch_data_tup[0]", torch_data_tup[0].shape)
    torch_data_np = torch_data_tup[0].detach().cpu().numpy().astype(np.float32)  # (B,T-1,D_torch)

    # ---- Shape sanity (D must match) ----
    if np_data.shape != torch_data_np.shape:
        return {
            "shapes_match": False,
            "numpy_shape": tuple(np_data.shape),
            "torch_shape": tuple(torch_data_np.shape),
            "note": "Feature dimensionalities differ; check offsets/chain/feet indices and the IK options.",
        }

    # ---- Metrics per batch ----
    diffs = np.abs(np_data[:, :, :] - torch_data_np[:, :, :])                   # (B,T-1,D)
    max_abs = diffs.reshape(B, -1).max(axis=1)                # (B,)
    mean_abs = diffs.reshape(B, -1).mean(axis=1)              # (B,)
    # np.allclose over flattened vectors per batch
    allclose = np.array([np.allclose(np_data[b], torch_data_np[b], atol=atol, rtol=rtol) for b in range(B)])

    #print(nan_report(np_data, torch_data_np))
    # Show where |diff|==1 (likely feet contacts)

    # Per-batch feet mismatches
    
    # Per-block MAE to verify only contacts differ
    blocks = {
        "root(4)": (0, 4),
        "ric": (4, 4 + (n_joints-1)*3),  # adjust n_joints if KIT
        "rot": (4 + (n_joints-1)*3, 4 + (n_joints-1)*3 + (n_joints-1)*6),
        "lvel": (4 + (n_joints-1)*3 + (n_joints-1)*6, 4 + (n_joints-1)*3 + (n_joints-1)*6 + n_joints*3),
        "feet(4)": ((4 + (n_joints-1)*3 + (n_joints-1)*6 + n_joints*3), (4 + (n_joints-1)*3 + (n_joints-1)*6 + n_joints*3 + 4)),
    }
    for name,(s,e) in blocks.items():
        mae = diffs[..., s:e].mean().item()
        max_ = diffs[..., s:e].max().item()
        print(name, mae, max_)

    result = {
        "shapes_match": True,
        "B": B,
        "T_minus_1": np_data.shape[1],
        "D": np_data.shape[2],
        "atol": atol,
        "rtol": rtol,
        "per_batch": [
            {"batch_idx": int(b), "max_abs_diff": float(max_abs[b]), "mean_abs_diff": float(mean_abs[b]), "allclose": bool(allclose[b])}
            for b in range(B)
        ],
        "allclose_overall": bool(allclose.all()),
    }

    if return_arrays:
        result["numpy_data"] = np_data
        result["torch_data"] = torch_data_np

    return result

sections = [
    0,
    4,
    4 + (n_joints-1)*3,
    4 + (n_joints-1)*3 + (n_joints-1)*6,
    4 + (n_joints-1)*3 + (n_joints-1)*6 + n_joints*3,
    4 + (n_joints-1)*3 + (n_joints-1)*6 + n_joints*3 + 4,
]
section_names = [
    "root",
    "ric",
    "rot",
    "lvel",
    "feet",
]
def meaning_full_index(index):
    for start, end, name in zip(sections[:-1], sections[1:], section_names):
        if index >= start and index < end:
            offset = index - start
            if name == "root":
                joint_idx = 0
                space_idx = ["rot_vel_y", "l_velocity_x", "l_velocity_z", "root_y"][offset]
            elif name == "ric":
                joint_idx = offset // 3 + 1
                space_idx = offset % 3
            elif name == "rot":
                joint_idx = offset // 6 + 1
                space_idx = offset % 6
            elif name == "lvel":
                joint_idx = offset // 3 + 1
                space_idx = offset % 3
            elif name == "feet":
                joint_idx = -1
                space_idx = offset
            return f"{name}_{joint_idx}_{space_idx}"
    return "Not Found"



if __name__ == "__main__":
    set_motion_process_module_globals()
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    motion_np_batch = []
    pad_length = 260
    min_length = 260
    check_hml_vecs_diffs = False # This spams prints
    for i in range(1):
        motion_vec_np = np.load(f"dataset/HumanML3D/new_joint_vecs/{i:06d}.npy")
        print("motion_vec_np", motion_vec_np.shape)
        motion_torch = recover_from_ric(torch.from_numpy(motion_vec_np).float(), n_joints)
        print("motion_torch", motion_torch.shape)
        if motion_torch.shape[0] < min_length:
            min_length = motion_torch.shape[0]
        motion_torch = torch.cat([motion_torch, torch.zeros((pad_length - motion_torch.shape[0], motion_torch.shape[1], motion_torch.shape[2]))], dim=0)
        motion_np_batch.append(motion_torch.numpy())
        if check_hml_vecs_diffs:
            motion_vec_torch = process_file_torch(motion_torch.unsqueeze(0))[0].squeeze(0)
            np_to_check = torch.from_numpy(motion_vec_np)[:-1, :]
            torch_to_check = motion_vec_torch[:np_to_check.shape[0], :]
            
            diffs = (torch_to_check - np_to_check).abs()
            for start, end, name in zip(sections[:-1], sections[1:], section_names):
                print(name, "mean", diffs[:, start:end].mean(dim=0))
                print(name, "max", diffs[:, start:end].max(dim=0).values)
                if name == "root":
                    print(name, "max indices", diffs[:, start:end].max(dim=0).indices)
            print("framewise diffs", "mean", diffs.mean(dim=1))
            print("framewise diffs", "max", diffs.max(dim=1).values)
            print("framewise diffs", "max indices", diffs.max(dim=1).indices)
            meaningful_indices = [meaning_full_index(index) for index in diffs.max(dim=1).indices]
            print("meaningful indices", meaningful_indices)
            xyz_torch = recover_from_ric(torch_to_check, n_joints)
            xyz_np = recover_from_ric(np_to_check, n_joints)
            diffs_xyz = (xyz_torch - xyz_np).abs()
            print("xyz diffs", "mean", diffs_xyz.mean(dim=0))
            print("xyz diffs", "max", diffs_xyz.max(dim=0).values)
    
    motion_np_batch = np.stack(motion_np_batch, axis=0)
    motion_np_batch = motion_np_batch[:, :min_length]

    result = compare_np_torch_features(
        motion_np_batch, 0.002, 
        paramUtil.t2m_raw_offsets, paramUtil.t2m_kinematic_chain, motion_process_module.face_joint_indx, 
        motion_process_module.fid_r, motion_process_module.fid_l
    )
    import json
    #print(json.dumps(result, indent=4))