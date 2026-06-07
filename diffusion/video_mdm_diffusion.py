import os
import numpy as np
import datetime
import torch
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from diffusion.losses import masked_l2
from utils.math_utils import perspective_projection_batch, sample_random_camera, perspective_projection_batch_angles
from data_loaders.humanml.scripts.motion_process import get_xyz_hunanml, recover_root_rot_pos
from diffusion.torch_process_motions import process_file_torch, mask_refactor_to_hmlvec, no_frame_loss_process_file_torch, get_skeleton_kwargs_from_dataset
from data_loaders.humanml.common.quaternion import qrot, qinv

#kinematic_tree_sources = []
#kinematic_tree_dests = []
#for chain in t2m_kinematic_chain:
#    kinematic_tree_sources.extend(chain[:-1])
#    kinematic_tree_dests.extend(chain[1:])
def motion_factors(motion_2d):
    vel_2d = motion_2d[..., 1:] - motion_2d[..., :-1]
    #join_vectors_2d = motion_2d[:, kinematic_tree_dests, ...] - motion_2d[:, kinematic_tree_sources, ...]
    #joint_lengths_2d = torch.sqrt(torch.sum(join_vectors_2d ** 2, dim=2, keepdim=True))
    #lengths_vel_2d = joint_lengths_2d[..., 1:] - joint_lengths_2d[..., :-1]
    #rotaions_2d = torch.atan2(join_vectors_2d[..., 1, :], join_vectors_2d[..., 0, :]).unsqueeze(2)
    return {
        "pos_2d":motion_2d, 
        "vel_2d":vel_2d, 
        #"joint_lengths_2d":joint_lengths_2d, 
        #"lengths_vel_2d":lengths_vel_2d, 
        #"rotaions_2d":rotaions_2d,
        #"joint_vectors_2d":join_vectors_2d,
    }


def norm_motion_2d(motion_2d: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
    """
    Normalize 2D motion sequences per sample using masked std and root first frame centering:
    - subtract root joint value (frame=0, joint=0)
    - compute std on raw seg = motion_2d[:length, :, 0/1]
    """
    device = motion_2d.device
    B, T, J, C = motion_2d.shape

    # mask valid time frames: (B, T, J, 1)
    idx = torch.arange(T, device=device)[None, :, None]
    mask = (idx < m_lens.to(device)[:, None, None]).unsqueeze(-1)
    mask = mask.expand(-1, -1, J, -1)  

    # subtract root joint at t=0, joint=0
    root = motion_2d[:, 0:1, 0:1, :]  # (B,1,1,2)
    x = motion_2d - root              # centered around root

    # masked mean of x per sample (B,1,1,2)
    count = mask.sum(dim=(1,2), keepdim=True)
    mean = (x * mask).sum(dim=(1,2), keepdim=True) / count

    # masked variance: E[(x-mean)^2]
    var = (((x - mean)**2) * mask).sum(dim=(1,2), keepdim=True) / count
    std = torch.sqrt(var).clamp(min=1e-6)

    # normalize
    normalized = -(x) / std

    # combine masked & original
    result = normalized * mask + motion_2d * (~mask)
    return result


def veolcity_to_2d(humanml_vec, camera, njoints):
    """
    delta_all = global_positions[:, 1:, :, :] - global_positions[:, :-1, :, :]     # (B, T-1, J, 3)
    r_rep_vel = r_rot[:, :-1, :].unsqueeze(-2).expand(-1, -1, J, -1)               # (B, T-1, J, 4)
    local_vel = qrot(r_rep_vel, delta_all).reshape(B, T-1, -1)                      # (B, T-1, J*3)

    This probably does not work, as this code is untested.
    """
    local_vel = humanml_vec[..., 4 + 9 * (njoints - 1) : 4 + 12 * (njoints - 1) + 3]
    r_rot_quat, r_pos = recover_root_rot_pos(humanml_vec)
    r_rot_quat = r_rot_quat.unsqueeze(-2).expand(-1, -1, njoints, -1)
    global_vel = qrot(qinv(r_rot_quat), local_vel)
    camera = camera.reshape(-1, 3, 4)
    camera_R = camera[:, :3, :3]
    camera_vel = camera_R.unsqueeze(1) @ global_vel.permute(0, 1, 3, 2)
    camera_vel = camera_vel.permute(0, 1, 3, 2)
    return camera_vel

def consistancy_hmlvec_loss(model_output, model_output_xyz, mask, mean, std, **kwargs):
    with torch.no_grad():
        consistant_model_output, _, _, _ = process_file_torch(model_output_xyz.detach().permute(0, 3, 1, 2), **kwargs)
        consistant_model_output = consistant_model_output.detach()
    consistant_model_output = (consistant_model_output - mean) / std

    # Reshape mask to hmlvec format
    mask = mask_refactor_to_hmlvec(mask, keep_shape=False, **kwargs)
    j_dim = model_output.shape[1]
    n_joints = int((j_dim + 1) // 12)
    pattern = torch.ones(j_dim, device=mask.device)

    cutoff_index = 4 + 3 * (n_joints - 1)
    pattern[:cutoff_index] = 0
    pattern = pattern.view(1, j_dim, 1, 1)
    mask = pattern * mask

    return masked_l2(model_output[..., :-1], consistant_model_output.permute(0, 2, 1).unsqueeze(2), mask)


def project_points_to_ray(
    points_3d: torch.Tensor,        # [B, N, 3]
    points_2d: torch.Tensor,        # [B, N, 2]  (u, v)
    camera_extrinsics: torch.Tensor, # [B, 3, 4]  ([R|t])
    orthographic: bool = False
) -> torch.Tensor:
    """
    For each batch item, project each 3D point onto the camera ray defined by (u,v,1)
    using the camera extrinsics [R|t], and return the closest point on that ray in
    *world* coordinates.

    Math:
      d = (u, v, 1)
      r = R^T d
      C = -R^T t
      λ = ( r·P̂ + d·t ) / ( d·d )
      P* = C + λ r
    Orthographic math:
      P̂_cam = R * P̂ + t
      P*_cam = (u, v, P̂_cam_z)
      P*_world = R^T * P*_cam - R^T * t
    """
    # ---- shape checks
    if points_3d.ndim != 3 or points_3d.size(-1) != 3:
        raise ValueError(f"points_3d must be [B, N, 3], got {tuple(points_3d.shape)}")
    if points_2d.ndim != 3 or points_2d.size(-1) != 2:
        raise ValueError(f"points_2d must be [B, N, 2], got {tuple(points_2d.shape)}")
    if camera_extrinsics.ndim != 3 or camera_extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"camera_extrinsics must be [B, 3, 4], got {tuple(camera_extrinsics.shape)}")

    Bp, Np, _ = points_3d.shape
    Bu, Nu, _ = points_2d.shape
    Be, _, _  = camera_extrinsics.shape
    if not (Bp == Bu == Be and Np == Nu):
        raise ValueError(f"Batch/N mismatch: points_3d {tuple(points_3d.shape)}, "
                         f"points_2d {tuple(points_2d.shape)}, extrinsics {tuple(camera_extrinsics.shape)}")

    # ---- extract R, t
    R = camera_extrinsics[..., :3, :3]                 # [B, 3, 3]
    t = camera_extrinsics[..., :3, 3]                  # [B, 3]
    RT = R.transpose(-1, -2)                           # [B, 3, 3]

    # ---- camera center C = -R^T t
    C = -torch.matmul(RT, t.unsqueeze(-1)).squeeze(-1) # [B, 3]

    if orthographic:
        points_3d_cam = points_3d @ RT + t.unsqueeze(1)
        projected_points_3d_cam = torch.cat([points_2d, points_3d_cam[:, :, 2:3]], dim=-1)
        projected_points_3d = projected_points_3d_cam @ R + C.unsqueeze(1)
        return projected_points_3d
    # ---- directions d = [u, v, 1]
    ones = torch.ones_like(points_2d[..., :1])         # [B, N, 1]
    D = torch.cat([points_2d, ones], dim=-1)           # [B, N, 3]

    # ---- world ray directions r = R^T d (batched matmul across all N at once)
    # D: [B, N, 3], RT: [B, 3, 3] -> r: [B, N, 3]
    r = torch.matmul(D, R)                        # [B, N, 3]  because D is a row vector, r = (R^T d^T)^T = d R

    # ---- denominator: d·d = u^2 + v^2 + 1
    denom = (D * D).sum(dim=-1)                        # [B, N]

    # ---- numerator: r·P̂ + d·t
    r_dot_P = (r * points_3d).sum(dim=-1)              # [B, N]
    d_dot_t = (D * t.unsqueeze(1)).sum(dim=-1)         # [B, N]
    lam = (r_dot_P + d_dot_t) / denom                  # [B, N]

    # ---- P* = C + λ r
    projected_points_3d = C.unsqueeze(1) + lam.unsqueeze(-1) * r   # [B, N, 3]
    return projected_points_3d


def project_points_to_ray_hmlvec(hmlvec: torch.Tensor, diffusion_extra: dict, hml_args: tuple, orthographic: bool = False, dataset=None) -> torch.Tensor:
    """
    B - batch size, D - vector dimension, J - number of joints, T - number of frames
    hmlvec [B, D, 1, T]
    diffusion_extra {'camera': [B, 12], 'motion_2d': [B, T, J, 2]}
    hml_args (mean, std)
    return [B, D, 1, T]
    """
    device = hmlvec.device
    # [B, T, J, 2] -> [B, J, 2, T]
    motion_2d = diffusion_extra['motion_2d'].to(device).permute(0, 2, 3, 1)
    pad_len = hmlvec.shape[-1] - motion_2d.shape[-1]
    motion_2d = torch.nn.functional.pad(motion_2d, (0, pad_len), value=0.0)  # pad time dimension
    B, J, _, T = motion_2d.shape
    xyz = get_xyz_hunanml(hmlvec, *hml_args) # This normlizes inside the function
    cameras = diffusion_extra['camera'].to(device).reshape(B, 3, 4)
    # [B, J, 2, T] -> [B, T, J, 2] -> [B, T*J, 2]
    points_2d = motion_2d.permute(0, 3, 1, 2).reshape(B, T*J, 2)
    # [B, J, 3, T] -> [B, T, J, 3] -> [B, T*J, 3]
    points_3d = xyz.permute(0, 3, 1, 2).reshape(B, T*J, 3)
    projected_points_3d = project_points_to_ray(points_3d, points_2d, cameras, orthographic=orthographic)
    # [B, T*J, 3] -> [B, T, J, 3]
    ret_xyz = projected_points_3d.reshape(B, T, J, 3)
    if dataset is not None:
        kwargs = get_skeleton_kwargs_from_dataset(dataset)
        ret_hmlvec_TD = no_frame_loss_process_file_torch(ret_xyz, **kwargs)
    else:
        ret_hmlvec_TD = no_frame_loss_process_file_torch(ret_xyz)
    # [B, T, D] -> [B, D, 1, T]
    ret_hmlvec = ret_hmlvec_TD.permute(0, 2, 1).unsqueeze(2)

    # normlized back per dataset standard
    mean, std = hml_args
    ret_hmlvec = (ret_hmlvec - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    return ret_hmlvec

def camera_0_to_1_hashing(camera, multiplier=99999):
    # camera is [B, 12] = [R(9), t(3)]
    # Simple hashing by summing all elements weighted by irregular weights, to get a value that is deterministic per camera but different across cameras.
    # Multiplying by 99999
    # Applying mod 1 function to get a value between 0 and 1.
    assert camera.ndim == 2 and camera.shape[1] == 12, "Camera should be of shape [B, 12]"
    irregular_weights = torch.tensor([0.9261, 0.4344, 0.5832, 0.8855,
                                      0.8350, 0.3663, 0.2924, 0.4221,
                                      0.0167, 0.9829, 0.2622, 0.1199
    ], device=camera.device, dtype=camera.dtype)
    hash_values = ((camera * irregular_weights).sum(dim=1) * multiplier) % 1
    return hash_values

def corrupt_camera(camera, azimuth_corruption):
    # camera is [B, 12] = [R(9), t(3)]
    R = camera[:, :9].reshape(-1, 3, 3)
    t = camera[:, 9:].reshape(-1, 3)

    # Sample azimuth corruption in radians
    camera_0_to_1_hashs = camera_0_to_1_hashing(camera)  # [B] value range reoughly between 0 and 1, deterministic per camera
    azimuth_noise = (camera_0_to_1_hashs - 0.5) * 2 * azimuth_corruption * np.pi / 180.0 # [B] value range between -azimuth_corruption and +azimuth_corruption in radians (assuming input is specified in degrees)

    # Create rotation matrix for azimuth corruption around the y-axis
    cos_a = torch.cos(azimuth_noise)
    sin_a = torch.sin(azimuth_noise)
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)
    R_azimuth = torch.stack([
        torch.stack([cos_a, zeros, sin_a], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_a, zeros, cos_a], dim=-1)
    ], dim=1)  # [B, 3, 3]

    # Apply azimuth corruption to the original rotation
    R_corrupted = R @ R_azimuth.transpose(-1, -2)  # [B, 3, 3]

    # Flatten back to [B, 12]
    corrupted_camera = torch.cat([R_corrupted.reshape(-1, 9), t.reshape(-1, 3)], dim=-1)
    return corrupted_camera


def calculate_video_mdm_loss(x_t, t, model_output, model_kwargs, mask, extra, terms, args, debug, distilation_model=None, terms_losses_prefix="", hml_args=None):
    loss = 0

    if args.train_camera_azimuth_corruption != 0:
        corrupted_camera = corrupt_camera(extra["camera"], args.train_camera_azimuth_corruption)
        original_cam = extra["camera"].clone()
        extra["camera"] = corrupted_camera

    model_output_xyz = get_xyz_hunanml(model_output, *hml_args)
    orthographic_projection = False #args.dataset in ["nba"]
    if args.lambda_floor_distance != 0:
        per_frame_min_height = model_output_xyz[:, :, 1, :].min(dim=1).values
        loss += args.lambda_floor_distance * (per_frame_min_height ** 2).mean(dim=1)
        terms[terms_losses_prefix + "floor_distance_loss"] = loss

    if args.gt_supervision:
        if args.gt_xyz_supervision:
            output_for_loss = model_output_xyz
            target_for_loss = get_xyz_hunanml(extra['gt_motion'].to(model_output_xyz.device), *hml_args)
        else:
            output_for_loss = model_output
            target_for_loss = extra['gt_motion'].to(model_output.device)
        mask_type = mask.dtype
        mask = (mask != 0).all(dim=1, keepdim=True).to(mask_type)
        terms[terms_losses_prefix + "gt_supervision_loss"] = masked_l2(output_for_loss, target_for_loss, mask)
        loss += terms[terms_losses_prefix + "gt_supervision_loss"]
        if debug:
            motion_2d = extra['motion_2d'].to(x_t.device)
            camera = extra['camera'].to(x_t.device)
            gt_motion = extra['gt_motion'].to(x_t.device)
            gt_xyz = get_xyz_hunanml(gt_motion, *hml_args)
            projected, distances = perspective_projection_batch(gt_xyz, camera, orthographic=orthographic_projection)
            projected = projected.permute(0, 3, 1, 2)
            terms["debug"]["model_output_xy"] = motion_2d.permute(0, 2, 3, 1)
            terms["debug"]["target_xy"] = projected.permute(0, 2, 3, 1)
            terms["debug"]["distances"] = distances      
    elif args.lambda_cam != 0:
        cameras = extra["camera"].to(model_output.device)
        #[batch_size, njoints, 2, nframes]
        target_xy = extra["motion_2d"].to(model_output.device).permute(0, 2, 3, 1)
        pad_len = mask.shape[-1] - target_xy.shape[-1]
        target_xy = torch.nn.functional.pad(target_xy, (0, pad_len), value=0.0)  # pad last dim
        model_output_xy, distances = perspective_projection_batch(model_output_xyz, cameras, orthographic=orthographic_projection)

        if orthographic_projection:
            DISTANCE_THRESHOLD = -9999
        elif args.no_distance_weighting:
            DISTANCE_THRESHOLD = 1e-6
        else:
            DISTANCE_THRESHOLD = 0.5 
        stable_distances = (distances > DISTANCE_THRESHOLD).expand(-1, -1, target_xy.shape[2], -1)
        def _where(a):
            return torch.where(stable_distances[:, -a.shape[1]:, :, -a.shape[3]:], a, torch.zeros_like(a))

        target_factors_xy = motion_factors(target_xy)
        model_output_factors_xy = motion_factors(model_output_xy)
        batch_size_distances = distances.sum(dim=(1,2,3)).squeeze() / (distances.shape[1] * distances.shape[3]) 
        if orthographic_projection or args.no_distance_weighting:
            batch_size_distances = torch.ones_like(batch_size_distances)
        terms["unstable_distances"] = (distances <= DISTANCE_THRESHOLD).sum().item()

        cam_losses = {
            key + "_mse": masked_l2(_where(target_val), _where(model_output_factors_xy[key]), mask[... ,-target_val.shape[-1]:]) * batch_size_distances.detach()
            for key, target_val in target_factors_xy.items()
        }

        terms[terms_losses_prefix + "cam_pos_loss"] = args.lambda_cam * cam_losses["pos_2d_mse"]
        terms[terms_losses_prefix + "cam_vel_loss"] = args.lambda_cam * args.lambda_cam_vel * cam_losses["vel_2d_mse"]

        loss += terms[terms_losses_prefix + "cam_pos_loss"] + terms[terms_losses_prefix + "cam_vel_loss"]
        if debug:
            terms["debug"]["model_output_xy"] = model_output_xy
            terms["debug"]["target_xy"] = target_xy
            terms["debug"]["distances"] = distances

    if distilation_model is not None:
        sds_losses = {}
        # Vectorised multi-camera SDS – duplicate the batch across `num_cameras_for_distilation` instead of looping.
        num_cams = args.num_cameras_for_distilation
        if num_cams > 1:
            model_output_xyz = model_output_xyz.repeat_interleave(num_cams, dim=0)
            x_t = x_t.repeat_interleave(num_cams, dim=0)
            t = t.repeat_interleave(num_cams)
            mask = mask.repeat_interleave(num_cams, dim=0)
            # Repeat `model_kwargs['y']` across the synthetic camera batch so all internal tensors keep the correct leading (batch) dimension. Works for both Tensor and dict-based structures coming from `tensors.py` / `dataset.py`.
            if 'y' in model_kwargs:
                y = model_kwargs['y']
                if torch.is_tensor(y):
                    model_kwargs['y'] = y.repeat_interleave(num_cams, dim=0)
                elif isinstance(y, dict):
                    y_rep = {}
                    for k, v in y.items():
                        if torch.is_tensor(v):
                            y_rep[k] = v.repeat_interleave(num_cams, dim=0)
                        elif isinstance(v, list):
                            # Duplicate each list element to preserve alignment with the expanded batch. Non-tensor lists (e.g. text strings) will have length `batch_size * num_cams` after replication.
                            y_rep[k] = [elem for elem in v for _ in range(num_cams)]
                        else:
                            y_rep[k] = v # For scalars or unsupported types, leave as-is.
                    model_kwargs['y'] = y_rep

        dbd = args.distilation_branched_denoising
        if not distilation_model.distilation_use_gt_camera:
            packed_values = sample_random_camera(model_output_xyz, distance_factor=args.cam_sample_distance_factor,
                                                min_cam_sample_elevation_angle=args.min_cam_sample_elevation_angle,
                                                max_cam_sample_elevation_angle=args.max_cam_sample_elevation_angle
            )
            cam_hor_angles, cam_ver_angles, cam_distances, shift = packed_values
            model_output_xy, distances = perspective_projection_batch_angles(model_output_xyz, cam_hor_angles, cam_ver_angles, cam_distances, shift,
                                                                            orthographic=dbd)

        elif args.lambda_cam == 0 or num_cams != 1: # Render only if didn't rander already for cam loss.
            cameras = extra["camera"].to(model_output.device)
            if num_cams > 1:
                cameras = cameras.repeat_interleave(num_cams, dim=0) if num_cams > 1 else cameras
            model_output_xy, distances = perspective_projection_batch(model_output_xyz, cameras,
                                                                                        orthographic=dbd)

        if dbd:
            xyz_t = get_xyz_hunanml(x_t, *hml_args)
            if not distilation_model.distilation_use_gt_camera:
                target_t, _ = perspective_projection_batch_angles(xyz_t, cam_hor_angles, cam_ver_angles, cam_distances, shift,
                                                                    orthographic=True)
            else:
                target_t, _ = perspective_projection_batch(xyz_t, cameras,
                                                                            orthographic=True)

        else:
            target_t = distilation_model.diffusion.q_sample(model_output_xy, t)
        # MAS CALL: model_output = model(x=x_t, timesteps=t, **cond)
        if not distilation_model.distilation_solve_ode:
            target_xy = distilation_model(x=target_t, timesteps=t, y=model_kwargs['y'])
        else:
            target_xy = distilation_model.diffusion.ddim_sample_loop(
                model=distilation_model,
                shape=target_t.shape,
                noise=target_t,  # Start from the noisy target_t
                clip_denoised=True,
                model_kwargs={'y': model_kwargs['y']},
                device=target_t.device,
                eta=0.0,  # Deterministic DDIM
                progress=False  # Set to True if you want to see progress
            )
        target_xy = target_xy.detach()

        target_factors_xy = motion_factors(target_xy)
        model_output_factors_xy = motion_factors(model_output_xy)
        batch_size_distances = distances.sum(dim=(1,2,3)).squeeze() / (distances.shape[1] * distances.shape[3])
        if dbd:
            batch_size_distances = torch.ones_like(batch_size_distances)
        DISTANCE_THRESHOLD = 0.5
        terms["unstable_distances"] = (distances <= DISTANCE_THRESHOLD).sum().item()
        stable_distances = (distances > DISTANCE_THRESHOLD).expand(-1, -1, target_xy.shape[2], -1)

        def _where(a):
            return torch.where(stable_distances[:, -a.shape[1]:, :, -a.shape[3]:], a, torch.zeros_like(a))

        for key, target_val in target_factors_xy.items():
            if key + "_mse" not in sds_losses:
                sds_losses[key + "_mse"] = 0
            # Average per-view loss across the camera dimension to recover original batch size.
            per_view_loss = masked_l2(
                _where(target_val),
                _where(model_output_factors_xy[key]),
                mask[..., -target_val.shape[-1]:]
            ) * batch_size_distances.detach()
            per_view_loss = per_view_loss.view(-1, num_cams).mean(dim=1)
            sds_losses[key + "_mse"] = per_view_loss

        terms[terms_losses_prefix + "sds_pos_loss"] = args.lambda_sds * sds_losses["pos_2d_mse"]
        terms[terms_losses_prefix + "sds_vel_loss"] = args.lambda_sds * args.lambda_sds_vel * sds_losses["vel_2d_mse"]
        loss += terms[terms_losses_prefix + "sds_pos_loss"] + terms[terms_losses_prefix + "sds_vel_loss"]

        if debug:
            if "model_output_xy" not in terms["debug"]:
                terms["debug"]["model_output_xy"] = model_output_xy
                terms["debug"]["target_xy"] = target_xy
                terms["debug"]["distances"] = distances
            else:
                terms["debug"]["model_output_xy_sds"] = model_output_xy
                terms["debug"]["target_xy_sds"] = target_xy
                terms["debug"]["distances_sds"] = distances
    
    if args.lambda_consistancy != 0:
        if args.gt_xyz_supervision:
            xyz = target_for_loss.detach()
        elif args.lambda_cam != 0:
            B, J, _, T = target_xy.shape
            cameras_ = cameras.reshape(B, 3, 4)
            # [B, J, 2, T] -> [B, T, J, 2] -> [B, T*J, 2]
            points_2d = target_xy.permute(0, 3, 1, 2).reshape(B, T*J, 2)
            # [B, J, 3, T] -> [B, T, J, 3] -> [B, T*J, 3]
            points_3d = model_output_xyz.permute(0, 3, 1, 2).reshape(B, T*J, 3)
            projected_points_3d = project_points_to_ray(points_3d, points_2d, cameras_, orthographic=orthographic_projection)
            # [B, T*J, 3] -> [B, J, 3, T]
            xyz = projected_points_3d.reshape(B, T, J, 3).permute(0, 2, 3, 1)
        else:
            raise ValueError("lambda_consistancy requires gt_xyz_supervision or lambda_cam != 0")
        chl = consistancy_hmlvec_loss(model_output, xyz, mask,
                                      *hml_args, **get_skeleton_kwargs_from_dataset(args.dataset))
        terms[terms_losses_prefix + "consistancy_hmlvec_loss"] = args.lambda_consistancy * chl
        loss += terms[terms_losses_prefix + "consistancy_hmlvec_loss"]
    
    if args.train_camera_azimuth_corruption != 0:
        extra["camera"] = original_cam

    return loss