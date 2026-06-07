import torch
from diffusion.video_mdm_diffusion import project_points_to_ray

# ---- Helpers to make extrinsics ----
def pack_extrinsics(R, t):
    # R: [3,3], t: [3]
    E = torch.zeros(3,4, dtype=R.dtype, device=R.device)
    E[:,:3] = R
    E[:, 3] = t
    return E

I = torch.eye(3)

# 1) Identity extrinsics: R=I, t=0, so C=0, r=d.
# Pick u=1, v=2 -> d=(1,2,1). Take a point exactly on the ray: P = 2*d.
B, N = 1, 1
R = I.clone(); t = torch.zeros(3)
E = pack_extrinsics(R, t)[None]                        # [1,3,4]
uv = torch.tensor([[[1., 2.]]])                        # [1,1,2]
P  = torch.tensor([[[2., 4., 2.]]])                    # [1,1,3]
out = project_points_to_ray(P, uv, E)
assert torch.allclose(out, P, atol=1e-6), "Identity case (on-ray) failed"

# 2) Pure translation: R=I, t=(0,0,-5) => C=(0,0,5).
# Ray for u=v=0 is along +Z, choose P=C+10*(0,0,1)=(0,0,15). Should be unchanged.
t = torch.tensor([0.,0.,-5.])
E = pack_extrinsics(I, t)[None]
uv = torch.tensor([[[0., 0.]]])
P  = torch.tensor([[[0., 0., 15.]]])
out = project_points_to_ray(P, uv, E)
assert torch.allclose(out, P, atol=1e-6), "Pure translation (on-ray) failed"

# 3) Off-ray point projects to the nearest line point
# Identity extrinsics, u=v=0, ray is z-axis. Point (3,4,7) should project to (0,0,7).
t = torch.zeros(3); E = pack_extrinsics(I, t)[None]
uv = torch.tensor([[[0., 0.]]])
P  = torch.tensor([[[3., 4., 7.]]])
out = project_points_to_ray(P, uv, E)
expected = torch.tensor([[[0., 0., 7.]]])
assert torch.allclose(out, expected, atol=1e-6), "Orthogonal projection to z-axis failed"

# 4) Shape error checks
try:
    project_points_to_ray(torch.zeros(1,2,4), torch.zeros(1,2,2), torch.zeros(1,3,4))
    raise AssertionError("Expected shape check to fail for points_3d last dim != 3")
except ValueError:
    pass

print("✅ Hard-coded tests passed.")

import numpy as np
import torch

# NumPy references (single example, non-batch)
def project_line_world_numpy(P, uv, R, t):
    """
    World-space formula: C = -R^T t, r = R^T d, λ = r·(P-C)/r·r, P* = C + λ r
    P: (3,), uv: (2,), R: (3,3), t: (3,)
    """
    d = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    C = -R.T @ t
    r = R.T @ d
    lam = r.dot(P - C) / r.dot(r)
    return C + lam * r

def project_via_camera_numpy(P, uv, R, t):
    """
    Camera-space formula: P_cam = R P + t; μ = (P_cam·d)/(d·d);
    P*_cam = μ d; P* = R^T (P*_cam - t)
    """
    d = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    P_cam = R @ P + t
    mu = P_cam.dot(d) / d.dot(d)
    P_star_cam = mu * d
    return R.T @ (P_star_cam - t)

def random_rotation(rng):
    A = rng.randn(3,3)
    Q, _ = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:,0] = -Q[:,0]
    return Q

def orthogonal_component(v, axis):
    # subtract projection onto axis
    return v - axis * (v.dot(axis) / axis.dot(axis))

def torch_from_np(x):
    return torch.tensor(x, dtype=torch.float64)

torch.set_default_dtype(torch.float64)

# Run many random trials
import random
SEED = 42
np.random.seed(SEED)                # global NumPy (optional if you never use it)
rng = np.random.RandomState(SEED)   # local NumPy RNG (preferred)
random.seed(SEED)                   # Python’s random
torch.manual_seed(SEED)             # PyTorch
torch.cuda.manual_seed_all(SEED)
rng = np.random.RandomState(42)
num_trials = 500
for _ in range(num_trials):
    R = random_rotation(rng)
    t = rng.randn(3) * 2.0

    uv = rng.randn(2)  # normalized coords (could also constrain)
    d  = np.array([uv[0], uv[1], 1.0])
    C  = -R.T @ t
    r  = R.T @ d

    # build a point near the ray: P = C + λ*r + e with e ⟂ r
    lam_true = rng.randn() * 3.0
    e_raw = rng.randn(3)
    e = orthogonal_component(e_raw, r) * 0.5  # small perpendicular noise
    P = C + lam_true * r + e

    # Reference answers
    P_ref1 = project_line_world_numpy(P, uv, R, t)
    P_ref2 = project_via_camera_numpy(P, uv, R, t)
    assert np.allclose(P_ref1, P_ref2, atol=1e-9)

    # Compare to torch implementation (B=N=1)
    E = np.zeros((3,4)); E[:,:3] = R; E[:,3] = t
    P_t  = torch_from_np(P)[None,None,:]         # [1,1,3]
    uv_t = torch_from_np(uv)[None,None,:]        # [1,1,2]
    E_t  = torch_from_np(E)[None,:,:]            # [1,3,4]
    out  = project_points_to_ray(P_t, uv_t, E_t).numpy()[0,0]

    assert np.allclose(out, P_ref1, atol=1e-7), f"Mismatch:\n{out}\n{P_ref1}"

print(f"✅ Random tests passed: {num_trials} trials.")
