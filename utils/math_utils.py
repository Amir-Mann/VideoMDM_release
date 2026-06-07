from math import sin
from math import cos
import numpy as np
import torch


def axis_rotation_matrix(angle, axis):
    if axis == "x":
        return torch.tensor([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]], dtype=torch.float32)
    elif axis == "y":
        return torch.tensor([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]], dtype=torch.float32)
    elif axis == "z":
        return torch.tensor([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]], dtype=torch.float32)

def rotation_matrix(angles):
    rotation = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    for axis, angle in angles:
        rotation = rotation @ axis_rotation_matrix(angle, axis)
    return rotation


def rotate(motion, angles):
    return motion @ rotation_matrix(angles).to(motion.device)


def rotate_multiple_angles(motion, hor_angles, ver_angles):
    return torch.stack([rotate(motion, [["y", hor_angle], ["x", ver_angle]]) for hor_angle, ver_angle in zip(hor_angles, ver_angles)], dim=0)


def perspective_projection_z_detatched(motion_3d, hor_angles, ver_angles, distance):
    rotated_motion_3d = rotate_multiple_angles(motion_3d, hor_angles, ver_angles)
    return rotated_motion_3d[..., :2] / (rotated_motion_3d[..., [2]].detach() + distance)


def perspective_projection(motion_3d, hor_angles, ver_angles, distance):
    rotated_motion_3d = rotate_multiple_angles(motion_3d, hor_angles, ver_angles)
    return rotated_motion_3d[..., :2] / (rotated_motion_3d[..., [2]] + distance)


def orthographic_projection(motion_3d, hor_angles, ver_angles):
    rotated_motion_3d = rotate_multiple_angles(motion_3d, hor_angles, ver_angles)
    return rotated_motion_3d[..., :2]


def dist(x, y):
    return torch.sum((x - y) ** 2, dim=-1)

def compute_into_camera_shift_from_angle(data, hor_angle):
    """ Find the shift required of the data so that looking at 0,0 with camera angle around y axis being hor_angle,
    The data would be infront of the camera wholy (for any none negative distance).
    data (torch.Tensor): Input data with dimensions [batch_size, njoints, 3, nframes].
    hor_angle torch.Tensor: The horizontal angle in radians with shape [batch_size].
    Returns: torch.Tensor: The shift of shape [batch_size, 3], containing (max_x, avg_y, max_z) for each item in thebatch."""

    x = data[:, :, 0, :]  # Shape: [batch_size, njoints, nframes]
    y = data[:, :, 1, :]  # Shape: [batch_size, njoints, nframes]
    z = data[:, :, 2, :]  # Shape: [batch_size, njoints, nframes]

    sin_hor = torch.sin (hor_angle)
    cos_hor = torch.cos(hor_angle)

    transformed_values = x * sin_hor[:, None, None] + z * cos_hor[:, None, None]  # Shape: [batch_size, njoints, nframes]

    #Here calculate the min_indecies which give the smallest per sample in batch, the index of the frame and joint closest to the camera.
    min_indices = transformed_values.reshape(transformed_values.shape[0], -1).argmin(dim=-1)  # Shape: [batch_size]

    # Convert flat indices back to original data dimensions
    min_joint_indices = min_indices // transformed_values.shape[2]# Joint indices
    min_frame_indices = min_indices % transformed_values.shape[2]  # Frame indices

    # Gather the corresponding min_x and min_z values
    min_x_values = x[torch.arange(data.shape[0]), min_joint_indices, min_frame_indices]
    min_z_values = z[torch.arange(data.shape[0]), min_joint_indices, min_frame_indices]

    avg_y = y.mean(dim=(1, 2))  # Average over joints and frames
    # The final shift is to set the motion to be right in front of the camera, except for the distance added separately.
    return - torch.stack([min_x_values, avg_y, min_z_values], dim=-1)  # Shape: [batch_size, 3]

def batch_axis_rotation_matrices(angles, axis):
    batch_size = angles.shape[0]
    if axis == "x":
        matrices = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device=angles.device)
        matrices[:, 0, 0] = 1
        matrices[:, 1, 1] = torch.cos(angles)
        matrices[:, 1, 2] = -torch.sin(angles)
        matrices[:, 2, 1] = torch.sin(angles)
        matrices[:, 2, 2] = torch.cos(angles)
    elif axis == "y":
        matrices = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device=angles.device)
        matrices[:, 1, 1] = 1
        matrices[:, 0, 0] = torch.cos(angles)
        matrices[:, 0, 2] = torch.sin(angles)
        matrices[:, 2, 0] = -torch.sin(angles)
        matrices[:, 2, 2] = torch.cos(angles)
    elif axis == "z":
        matrices = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device=angles.device)
        matrices[:, 2, 2] = 1
        matrices[:, 0, 0] = torch.cos(angles)
        matrices[:, 0, 1] = -torch.sin(angles)
        matrices[:, 1, 0] = torch.sin(angles)
        matrices[:, 1, 1] = torch.cos(angles)
    else:
        raise ValueError(f"Invalid axis: {axis}")
    return matrices

def batch_rotation_matrix(hor_angles, ver_angles):
    """Generates a combined batch rotation matrix."""
    y_rot = batch_axis_rotation_matrices(hor_angles, "y")
    x_rot = batch_axis_rotation_matrices(ver_angles, "x")
    return torch.bmm(y_rot, x_rot)

def perspective_projection_batch_angles(motion_3d, hor_angles, ver_angles, cam_distances, shift=None, distance_stability=1e-6, orthographic=False):
    """
    ret: Tensor of shape [batch_size, njoints, 2, nframes]
    motion_3d (torch.Tensor): Input 3D motion tensor of shape [batch_size, njoints, 3, nframes].
    hor_angles (torch.Tensor): Horizontal rotation angles of shape [batch_size].
    ver_angles (torch.Tensor): Vertical rotation angles of shape [batch_size].
    cam_distances (torch.Tensor): Distances to the camera of shape [batch_size].
    shift (torch.Tensor or None): Optional tensor of shape [batch_size, 3] for per-sample shift.
    distance_stability (float): Stability factor for distance computation.
    orthographic (bool): Whether to use orthographic projection.
    """
    # Reshape and compute rotation matrices
    rotation_matrices = batch_rotation_matrix(hor_angles, ver_angles).unsqueeze(1)  # Shape: [batch_size, 1, 3, 3]

    # Rotate the motion data
    motion_3d = motion_3d.permute(0, 3, 1, 2)  # [batch_size, nframes, njoints, 3]
    if shift is not None:
        shift = shift.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, 3]
        motion_3d = motion_3d + shift  # Apply shift

    rotated_motion = torch.matmul(motion_3d, rotation_matrices).permute(0, 2, 3, 1)  # [batch_size, njoints, 3, nframes]

    # Compute perspective projection
    cam_distances = cam_distances.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
    distances_tensor = rotated_motion[..., [2], :] + cam_distances
    distances_tensor = torch.abs(distances_tensor) + distance_stability
    if not orthographic:
        proj_2d = rotated_motion[..., :2, :] / (distances_tensor.detach())  # [batch_size, njoints, 2, nframes]
    else:
        proj_2d = rotated_motion[..., :2, :]

    return proj_2d, distances_tensor

def sample_random_camera(data, distance_factor=2.5, min_cam_sample_elevation_angle=-np.pi/8, max_cam_sample_elevation_angle=np.pi/24,
                         return_matrix=False, use_hor_angle=None):
    batch_size = data.shape[0]
    cam_hor_angles = torch.rand(batch_size, device=data.device) * 2 * np.pi  # [0, 2*pi]
    if use_hor_angle is not None:
        cam_hor_angles = use_hor_angle * torch.ones(batch_size, device=data.device)
    start, end = min_cam_sample_elevation_angle, max_cam_sample_elevation_angle
    cam_ver_angles = torch.rand(batch_size, device=data.device) * (end - start) + start 

    cam_distances = torch.ones(batch_size, device=data.device) * distance_factor
    shift = compute_into_camera_shift_from_angle(data, cam_hor_angles)
    if return_matrix:
        R = batch_rotation_matrix(cam_hor_angles, cam_ver_angles).permute(0, 2, 1) # [batch_size, 3, 3]
        # [batch_size, 3, 1]
        dist_shift = R.permute(0, 2, 1) @ torch.tensor([0, 0, 1], dtype=data.dtype, device=data.device).unsqueeze(0).unsqueeze(-1) * cam_distances.unsqueeze(-1).unsqueeze(-1)
        dist_shift = dist_shift.squeeze(2) # [batch_size, 3]
        minus_C = (shift) + dist_shift
        t = R @ minus_C.unsqueeze(-1)
        return torch.cat([R, t], dim=-1) # [batch_size, 3, 4]
    else:
        return cam_hor_angles, cam_ver_angles, cam_distances, shift

def perspective_projection_batch(
    motion_3d: torch.Tensor,        # [B, J, 3, T]
    cams: torch.Tensor,             # [B, 12]
    distance_stability: float = 1e-6,
    orthographic: bool = False
):
    B, J, _, T = motion_3d.shape
    assert cams.shape == (B, 12), f"cams must be of shape [B, 12] but got {cams.shape} with B={B}"
    assert motion_3d.shape == (B, J, 3, T), f"motion_3d must be of shape [B, J, 3, T] but got {motion_3d.shape} with B={B}, J={J}, 3=3, T={T}"

    # Batch multiplication (B, 1, 3, 4) @ (B, J, 4, T) -> (B, J, 3, T)
    ones = torch.ones(B, J, 1, T, device=motion_3d.device, dtype=motion_3d.dtype)
    motion_cam = cams.view(B, 1, 3, 4) @ torch.cat([motion_3d, ones], dim=2)

    # Separate depth (z) and project
    depths_camera = motion_cam[..., 2:, :]              # [B, J, 1, T]
    depths_camera = depths_camera.abs() + distance_stability

    if not orthographic:
        motion_cam = motion_cam[..., :2, :] / depths_camera                      # perspective division
    else:
        motion_cam = motion_cam[..., :2, :]

    return motion_cam, depths_camera


def quaternion_mul(q, r):
    """Multiply two quaternions q⊗r, both shape (4,) in (x,y,z,w) order."""
    x1, y1, z1, w1 = q.unbind(0)
    x2, y2, z2, w2 = r.unbind(0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((x,y,z,w), dim=0)

def optimize_camera_params_and_scale(
    xyz_data, motion_2d, f, camera_initial,
    num_iters=1000, lr=1e-3, device=None
):
    """
    xyz_data:       np array (N, J, 3)
    motion_2d:      np array (N, J, 2)
    f:              np array (N, J)      -- weights
    camera_initial: np array (7,) = [x,y,z, qx,qy,qz,qw]
    returns: scale_x, scale_y, camera (7,) as np
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fixed initial values
    init_pos  = torch.from_numpy(camera_initial[:3]).float().to(device)
    init_quat = torch.from_numpy(camera_initial[3:]).float().to(device)
    init_quat = init_quat / init_quat.norm(p=2)

    # learnable elevation and tilt
    cam_y_p    = torch.nn.Parameter(init_pos[1].clone())
    tilt_angle = torch.nn.Parameter(torch.tensor(0.0, device=device))

    # prepare data
    N, J, _    = xyz_data.shape
    motion_3d_t = (torch.from_numpy(xyz_data)
                       .float()
                       .permute(1,2,0)   # [J,3,N]
                       .unsqueeze(0)     # [1,J,3,N]
                       .to(device))
    target_2d  = torch.from_numpy(motion_2d).float().to(device)      # [N,J,2]
    weights    = torch.from_numpy(f).float().to(device).unsqueeze(-1) # [N,J,1]

    optim      = torch.optim.Adam([cam_y_p, tilt_angle], lr=lr)
    flip       = torch.tensor([-1.0,1.0], device=device).view(1,1,2)

    loop_th = range(num_iters)
    if num_iters > 100:
        from tqdm import tqdm
        loop_th = tqdm(loop_th, desc="Camera Optimization", total=num_iters)
    for _ in loop_th:
        optim.zero_grad()

        # rebuild camera
        cam_pos     = init_pos.clone()
        cam_pos[1]  = cam_y_p
        half        = tilt_angle / 2
        q_tilt      = torch.cat([
                         torch.sin(half).expand(3) * torch.tensor([1,0,0], device=device),
                         torch.cos(half).view(1)
                     ], dim=0)            # [4]
        q           = quaternion_mul(q_tilt, init_quat)
        cams        = torch.cat([cam_pos, q], dim=0).unsqueeze(0)  # [1,7]

        # project
        proj, _     = perspective_projection_batch_pos_and_rotation(motion_3d_t, cams)
        proj        = proj.permute(0,3,1,2).squeeze(0) * flip       # [N,J,2]

        # compute scales (detached)
        sx = (target_2d[...,0].std() / proj[...,0].std()).detach()
        sy = (target_2d[...,1].std() / proj[...,1].std()).detach()

        # scale & align observed
        obs_x = target_2d[...,0:1] / sx
        obs_y = target_2d[...,1:2] / sy
        obs   = torch.cat([obs_x, obs_y], dim=-1)                   # [N,J,2]
        pivot_obs = obs[0,0]                                        # [2]
        pivot_prj = proj[0,0]                                       # [2]
        obs   = obs - pivot_obs.view(1,1,2) + pivot_prj.view(1,1,2)

        loss  = ((obs - proj).pow(2) * weights).mean() + (cam_y_p - init_pos[1]).pow(2)
        loss.backward()
        optim.step()

    # final camera
    cam_pos     = init_pos.clone()
    cam_pos[1]  = cam_y_p.detach()
    half        = tilt_angle.detach() / 2
    q_tilt      = torch.cat([
                     torch.sin(half).expand(3) * torch.tensor([1,0,0], device=device),
                     torch.cos(half).view(1)
                 ], dim=0)
    cam_quat    = quaternion_mul(q_tilt, init_quat).cpu()
    camera      = torch.cat([cam_pos.cpu(), cam_quat], dim=0).numpy()

    # final projection for scale return
    with torch.no_grad():
        cams   = torch.cat([cam_pos, cam_quat.to(device)], dim=0).unsqueeze(0)
        proj_f,_ = perspective_projection_batch_pos_and_rotation(motion_3d_t, cams)
        proj_f   = proj_f.permute(0,3,1,2).squeeze(0) * flip
        scale_x  = (target_2d[...,0].std() / proj_f[...,0].std()).item()
        scale_y  = (target_2d[...,1].std() / proj_f[...,1].std()).item()

    return scale_x, scale_y, camera
# ====================== quaternions ======================


def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return qinv(torch.from_numpy(q).float()).numpy()


def qnormalize(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    return q / torch.norm(q, dim=-1, keepdim=True)


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0, deg=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    if deg:
        return torch.stack((x, y, z), dim=1).view(original_shape) * 180 / np.pi
    else:
        return torch.stack((x, y, z), dim=1).view(original_shape)


# Numpy-backed implementations


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda().float()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous().float()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def euler2quat(e, order, deg=True):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.view(-1, 3)

    ## if euler angles in degrees
    if deg:
        e = e * np.pi / 180.0

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.view(original_shape)


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def cont6d_to_matrix_np(cont6d):
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


def qpow(q0, t, dtype=torch.float):
    """q0 : tensor of quaternions
    t: tensor of powers
    """
    q0 = qnormalize(q0)
    theta0 = torch.acos(q0[..., 0])

    ## if theta0 is close to zero, add epsilon to avoid NaNs
    mask = (theta0 <= 10e-10) * (theta0 >= -10e-10)
    theta0 = (1 - mask) * theta0 + mask * 10e-10
    v0 = q0[..., 1:] / torch.sin(theta0).view(-1, 1)

    if isinstance(t, torch.Tensor):
        q = torch.zeros(t.shape + q0.shape)
        theta = t.view(-1, 1) * theta0.view(1, -1)
    else:  ## if t is a number
        q = torch.zeros(q0.shape)
        theta = t * theta0

    q[..., 0] = torch.cos(theta)
    q[..., 1:] = v0 * torch.sin(theta).unsqueeze(-1)

    return q.to(dtype)


def qslerp(q0, q1, t):
    """
    q0: starting quaternion
    q1: ending quaternion
    t: array of points along the way

    Returns:
    Tensor of Slerps: t.shape + q0.shape
    """

    q0 = qnormalize(q0)
    q1 = qnormalize(q1)
    q_ = qpow(qmul(q1, qinv(q0)), t)

    return qmul(q_, q0.contiguous().view(torch.Size([1] * len(t.shape)) + q0.shape).expand(t.shape + q0.shape).contiguous())


def qbetween(v0, v1):
    """
    find the quaternion used to rotate v0 to v1
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0**2).sum(dim=-1, keepdim=True) * (v1**2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0, v1):
    """
    find the quaternion used to rotate v0 to v1
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


def lerp(p0, p1, t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor([t])

    new_shape = t.shape + p0.shape
    new_view_t = t.shape + torch.Size([1] * len(p0.shape))
    new_view_p = torch.Size([1] * len(t.shape)) + p0.shape
    p0 = p0.view(new_view_p).expand(new_shape)
    p1 = p1.view(new_view_p).expand(new_shape)
    t = t.view(new_view_t).expand(new_shape)

    return p0 + t * (p1 - p0)
