import torch
import torch.nn.functional as F

IdxLike = int | slice | torch.Tensor
def select_plane(vector:torch.Tensor, plane_index:IdxLike|None=None)->torch.Tensor:
    if plane_index is None:
        return vector
    
    return vector[..., plane_index]

def compute_distance(
    pos_from: torch.Tensor,
    pos_to: torch.Tensor,
    offset: torch.Tensor|None = None,
    plane_index:IdxLike|None=None,
) -> torch.Tensor:
    """
    Euclidean distance between pos_from and pos_to (supports batching).
    pos_* assumed shape (..., 3). 'plane_index' can be an int or a list like [0,1] to work in a plane.
    """
    if offset is not None:
        pos_to = pos_to + offset
    pos_from = select_plane(pos_from, plane_index)
    pos_to = select_plane(pos_to, plane_index)
    return torch.norm(pos_to - pos_from, p=2, dim=-1)

def compute_direction(
    pos_from: torch.Tensor,
    pos_to: torch.Tensor,
    offset: torch.Tensor|None = None,
    plane_index: IdxLike|None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Unit vector from pos_from to pos_to (supports batching).
    Always divides by at least `eps` to avoid NaN/Inf.
    """
    if offset is not None:
        pos_to = pos_to + offset

    pos_from = select_plane(pos_from, plane_index)
    pos_to = select_plane(pos_to, plane_index)

    d = pos_to - pos_from
    n = torch.norm(d, dim=-1, keepdim=True)
    return d / (n + eps) 

def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize quaternion(s) to unit length. q shape (..., 4) with (w, x, y, z) or (x,y,z,w) — pick one & stick to it.
    Below assumes (w, x, y, z). If you store as (x,y,z,w), swap accordingly.
    """
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_shortest_dot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Dot product with double-cover handled: returns |<q1, q2>| so that q and -q are identical.
    Shape: (...,).
    """
    q1 = _quat_normalize(q1)
    q2 = _quat_normalize(q2)
    return torch.clamp(torch.sum(q1 * q2, dim=-1).abs(), max=1.0)

def quat_geodesic_angle(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Smallest angle (in radians) between orientations q1 and q2 on SO(3).
    Angle = 2 * arccos(|dot(q1, q2)|).
    Returns shape (...,).
    """
    d = quat_shortest_dot(q1, q2)
    # numerical clamp to valid domain
    d = torch.clamp(d, -1.0 + eps, 1.0 - eps)
    return 2.0 * torch.arccos(d)

def is_quat_reached(
    q: torch.Tensor,
    q_target: torch.Tensor,
    threshold_rad: float,
) -> torch.Tensor:
    """
    True if the orientation is within threshold_rad (radians) of target.
    q, q_target: shape (..., 4)
    returns: shape (...,) boolean tensor
    """
    ang = quat_geodesic_angle(q, q_target)
    return ang <= threshold_rad

def is_rotating_to_quat(
    q_from: torch.Tensor,
    q_to: torch.Tensor,
    q_target: torch.Tensor,
    tol_decrease: float = 1e-4,
) -> torch.Tensor:
    """
    True if between t and t+1 we got closer to target orientation by at least tol_decrease radians.
    Uses geodesic angle monotonic decrease (robust & sign-free).
    q_* shape (..., 4); returns (...,) boolean tensor.
    """
    a_from = quat_geodesic_angle(q_from, q_target)
    a_to   = quat_geodesic_angle(q_to,   q_target)
    return (a_from - a_to) >= tol_decrease

class MotionDetector:
    @staticmethod
    def is_aligned(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        threshold: float,
        plane_index: IdxLike,
        offset: torch.Tensor|None = None,
    ) -> torch.Tensor:
        # same behavior, cleaner implementation
        d = compute_distance(source_pos, target_pos,
                             offset=offset, plane_index=plane_index)
        return d <= threshold
    
    @staticmethod
    def is_above(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        offset: float = 0.0,
        aligned: bool = False,
        xy_align_tol: float = 0.05,
        z_band: float = 0.05,
    ) -> torch.Tensor:
        xy_d = compute_distance(source_pos, target_pos, plane_index=[0, 1])
        is_xy_aligned = xy_d <= xy_align_tol
        zs = source_pos[..., 2]
        zt = target_pos[..., 2] + offset
        is_above = (zs > zt) & (zs <= (zt + z_band))
        return is_xy_aligned & is_above if aligned else is_above
    
    @staticmethod
    def is_below(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        offset: float = 0.0,
        aligned: bool = False,
        xy_align_tol: float = 0.025,
    ) -> torch.Tensor:
        xy_d = compute_distance(source_pos, target_pos, plane_index=(0, 1))
        is_xy_aligned = xy_d <= xy_align_tol
        zs = source_pos[..., 2]
        zt = target_pos[..., 2] + offset
        is_below = zs < zt
        return is_xy_aligned & is_below if aligned else is_below

    @staticmethod
    def is_left(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        offset: float = 0.0,
        aligned: bool = False,
        yz_align_tol: float = 0.05,
    ) -> torch.Tensor:
        yz_d = compute_distance(source_pos, target_pos, plane_index=(1, 2))
        is_yz_aligned = yz_d <= yz_align_tol
        xs = source_pos[..., 0]
        xt = target_pos[..., 0] + offset
        is_left = xs < xt
        return is_yz_aligned & is_left if aligned else is_left

    @staticmethod
    def is_right(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        offset: float = 0.0,
        aligned: bool = False,
        yz_align_tol: float = 0.05,
    ) -> torch.Tensor:
        yz_d = compute_distance(source_pos, target_pos, plane_index=(1, 2))
        is_yz_aligned = yz_d <= yz_align_tol
        xs = source_pos[..., 0]
        xt = target_pos[..., 0] + offset
        is_right = xs > xt
        return is_yz_aligned & is_right if aligned else is_right

    @staticmethod
    def is_front(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        offset: float = 0.0,
        aligned: bool = False,
        xz_align_tol: float = 0.05,
    ) -> torch.Tensor:
        xz_d = compute_distance(source_pos, target_pos, plane_index=(0, 2))
        is_xz_aligned = xz_d <= xz_align_tol
        ys = source_pos[..., 1]
        yt = target_pos[..., 1] + offset
        is_front = ys < yt
        return is_xz_aligned & is_front if aligned else is_front

    @staticmethod
    def is_back(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        offset: float = 0.0,
        aligned: bool = False,
        xz_align_tol: float = 0.05,
    ) -> torch.Tensor:
        xz_d = compute_distance(source_pos, target_pos, plane_index=(0, 2))
        is_xz_aligned = xz_d <= xz_align_tol
        ys = source_pos[..., 1]
        yt = target_pos[..., 1] + offset
        is_back = ys > yt
        return is_xz_aligned & is_back if aligned else is_back

    @staticmethod
    def is_moving_to(
        source_pos_from: torch.Tensor,
        source_pos_to: torch.Tensor,
        target_pos: torch.Tensor,
        threshold: float,
        offset: torch.Tensor|None = None,
        plane_index: IdxLike|None = None,
    ) -> torch.Tensor:
        """
        Returns True if the motion direction (source_t+1 - source_t) has cosine similarity
        to the goal direction (target - source_t [+ offset]) >= threshold.
        """
        move_dir = compute_direction(source_pos_from, source_pos_to, plane_index=plane_index)
        goal_dir = compute_direction(source_pos_from, target_pos, offset=offset, plane_index=plane_index)
        cos = F.cosine_similarity(move_dir, goal_dir, dim=-1)
        return cos >= threshold

    @staticmethod
    def is_reached(
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        threshold: float,
        offset: torch.Tensor|None = None,
    ) -> torch.Tensor:
        d = compute_distance(source_pos, target_pos, offset=offset)
        return d <= threshold
    
    @staticmethod
    def is_closing_distance(gripper_distance_from: torch.Tensor,
                    gripper_distance_to: torch.Tensor,
                    min_close: float = 0.0025) -> torch.Tensor:
        """
        True if finger distance decreases by at least 'min_close'.
        """
        return (gripper_distance_from - gripper_distance_to) >= min_close
    
    @staticmethod
    def is_closing_joint_pos(gripper_distance_from: torch.Tensor,
                    gripper_distance_to: torch.Tensor,
                    min_close: float = 0.0025,
                    reversed: bool = False) -> torch.Tensor:
        """
        True if finger distance decreases by at least 'min_close'.
        """
        if reversed:
            return (gripper_distance_to - gripper_distance_from) >= min_close
        
        return (gripper_distance_from - gripper_distance_to) >= min_close
    
    @staticmethod
    def is_closing_distance(gripper_distance_from: torch.Tensor,
                    gripper_distance_to: torch.Tensor,
                    min_close: float = 0.0025) -> torch.Tensor:
        """
        True if finger distance decreases by at least 'min_close'.
        """
        return (gripper_distance_to - gripper_distance_from) >= min_close
    
    @staticmethod
    def is_openning_joint_pos(gripper_distance_from: torch.Tensor,
                    gripper_distance_to: torch.Tensor,
                    min_close: float = 0.0025,
                    reversed: bool = False) -> torch.Tensor:
        """
        True if finger distance decreases by at least 'min_close'.
        """
        if reversed:
            return (gripper_distance_from - gripper_distance_to) >= min_close
        
        return (gripper_distance_to - gripper_distance_from) >= min_close
    
    @staticmethod
    def is_gripper_closed(gripper_joint_pos: torch.Tensor, threshold: float) -> torch.Tensor:
        return gripper_joint_pos <= threshold
    
    @staticmethod
    def is_gripper_opened(gripper_joint_pos: torch.Tensor, threshold: float) -> torch.Tensor:
        return gripper_joint_pos >= threshold
    
    @staticmethod
    def is_moving(
        pos: torch.Tensor,
        next_pos: torch.Tensor,
        threshold: float,
        plane_index: IdxLike = None,
    ) -> torch.Tensor:
        d = compute_distance(pos, next_pos, plane_index=plane_index)
        return d >= threshold
    
    @staticmethod
    def is_rotating_to_quat(
        quat_from: torch.Tensor,
        quat_to: torch.Tensor,
        target_quat: torch.Tensor,
        tol_decrease: float = 1e-4,
    ) -> torch.Tensor:
        """
        Returns True if orientation moves closer to target (by >= tol_decrease radians).
        """
        return is_rotating_to_quat(quat_from, quat_to, target_quat, tol_decrease=tol_decrease)

    @staticmethod
    def is_quat_reached(
        quat: torch.Tensor,
        target_quat: torch.Tensor,
        threshold_rad: float = 2.5 * torch.pi / 180.0,  # ~2.5 degrees default
    ) -> torch.Tensor:
        """
        Returns True if orientation is within threshold_rad of target.
        """
        return is_quat_reached(quat, target_quat, threshold_rad)
