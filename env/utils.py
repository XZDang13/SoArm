import torch
import math

def quat_to_rotmat(q: torch.Tensor, xyzw: bool = True) -> torch.Tensor:
    if xyzw:
        x, y, z, w = q.unbind(-1)
    else:
        w, x, y, z = q.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    R[...,0,0] = 1 - 2*(yy + zz)
    R[...,0,1] = 2*(xy - wz)
    R[...,0,2] = 2*(xz + wy)
    R[...,1,0] = 2*(xy + wz)
    R[...,1,1] = 1 - 2*(xx + zz)
    R[...,1,2] = 2*(yz - wx)
    R[...,2,0] = 2*(xz - wy)
    R[...,2,1] = 2*(yz + wx)
    R[...,2,2] = 1 - 2*(xx + yy)
    return R

def yaw_from_rotmat_robust(R: torch.Tensor) -> torch.Tensor:
    
    a, b = R[...,0,0], R[...,0,1]
    c, d = R[...,1,0], R[...,1,1]
    return torch.atan2(c - b, a + d)

def quat_from_yaw(yaw: torch.Tensor, xyzw: bool = True) -> torch.Tensor:
    half = 0.5 * yaw
    cz, sz = torch.cos(half), torch.sin(half)
    x = torch.zeros_like(cz)
    y = torch.zeros_like(cz)
    z = sz
    w = cz
    q = torch.stack([x, y, z, w], dim=-1) if xyzw else torch.stack([w, x, y, z], dim=-1)
    
    sign = torch.where(q[..., -1:] < 0, -1.0, 1.0)
    return q * sign

@torch.no_grad()
def map_to_yaw_rep(q: torch.Tensor, xyzw: bool = True) -> torch.Tensor:
    R = quat_to_rotmat(q, xyzw=xyzw)
    yaw = yaw_from_rotmat_robust(R)
    return quat_from_yaw(yaw, xyzw=xyzw)

def add_quat_noise_wxyz(quat: torch.Tensor, max_angle: float = 0.05) -> torch.Tensor:
    assert quat.shape[-1] == 4, "quat must have shape (..., 4)"

    # Random axis (uniform on sphere)
    axis = torch.randn_like(quat[..., 1:4])
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)

    # Random rotation angle from uniform[-max_angle, max_angle]
    angle = (torch.rand(quat.shape[:-1], device=quat.device) * 2 - 1) * max_angle
    half_angle = 0.5 * angle

    # Axis-angle → quaternion [w, x, y, z]
    delta_quat = torch.cat([
        torch.cos(half_angle)[..., None],
        axis * torch.sin(half_angle)[..., None]
    ], dim=-1)

    # Quaternion multiplication: q_noisy = delta * quat
    w1, x1, y1, z1 = delta_quat.unbind(-1)
    w2, x2, y2, z2 = quat.unbind(-1)
    noisy = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

    # Normalize
    noisy = noisy / (torch.norm(noisy, dim=-1, keepdim=True) + 1e-8)
    return noisy
