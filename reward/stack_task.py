import torch
from .motion_detector import MotionDetector, quat_geodesic_angle

class StackTaskReward:
    @staticmethod
    def compute_reward(gripper_pos_from: torch.Tensor,
                       gripper_quat_from: torch.Tensor,
                       gripper_pos_to: torch.Tensor,
                       gripper_quat_to: torch.Tensor,
                       target_pos: torch.Tensor,
                       target_quat: torch.Tensor,
                       gripper_joint_pos_from: torch.Tensor,
                       gripper_joint_pos_to: torch.Tensor,
                       is_grasped: torch.Tensor,
                       is_debug: bool) -> torch.Tensor:
        """Staged reward for stack task: 
        1) move above cube (prep) 
        2) descend to cube (approach)
        3) contact and close (grasp) 
        4) hold still (stabilize)
        Returns a scalar per-env reward (same shape as is_grasped).
        """
        # Tunables
        z_band = 0.15                   # meters regarded as "above"
        xy_align_tol = 0.015            # lateral alignment tolerance for 'above'
        move_to_cos = 0.5              # cosine threshold for 'moving toward'
        reach_tol = 0.02               # m distance to count as reached
        quat_align_deg = 25.0           # degrees for alignment
        quat_align_rad = quat_align_deg * torch.pi / 180.0
        quat_improve_tol = 0.003        # rad improvement considered as "rotating to"
        move_noise_tol = 0.002          # 'moving' magnitude to consider non-jitter
        open_abs_thresh = torch.pi / 4  # absolute openness considered 'open enough' pre-grasp
        open_delta = 0.003              # active opening min delta
        close_delta = 0.003             # active closing min delta

        # Derived targets
        # "top" position: directly above target by z offset
        top_offset = torch.zeros_like(target_pos)
        top_offset[..., 2] = z_band  # reuse z_band as nominal vertical offset
        target_top = target_pos + top_offset

        # Signals
        is_above = MotionDetector.is_above(
            gripper_pos_to, target_pos, z_band=z_band, xy_align_tol=xy_align_tol
        )

        is_moving_to_top = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to, target_top, threshold=move_to_cos
        )

        is_moving_to_target = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to, target_pos, threshold=move_to_cos
        )

        is_reached = MotionDetector.is_reached(
            gripper_pos_to, target_pos, threshold=reach_tol
        )

        is_moving = MotionDetector.is_moving(
            gripper_pos_from, gripper_pos_to, threshold=move_noise_tol
        )

        is_rotating_to = MotionDetector.is_rotating_to_quat(
            gripper_quat_from, gripper_quat_to, target_quat, tol_decrease=quat_improve_tol
        )

        is_quat_aligned = MotionDetector.is_quat_reached(
            gripper_quat_to, target_quat, threshold_rad=quat_align_rad
        )

        # Opening/closing semantics
        actively_opening = MotionDetector.is_opening_joint_pos(
            gripper_joint_pos_from, gripper_joint_pos_to, min_close=open_delta
        )
        pregrasp_open_ok = MotionDetector.is_gripper_opened(
            gripper_joint_pos_to, open_abs_thresh
        )
        is_gripper_opening = actively_opening | ((~is_above) & pregrasp_open_ok)

        is_gripper_closing = MotionDetector.is_closing_joint_pos(
            gripper_joint_pos_from, gripper_joint_pos_to, min_close=close_delta
        )

        stage_1 = (is_moving_to_top & (is_quat_aligned | is_rotating_to) &
                   is_gripper_opening & is_moving & (~is_above)).float() * 0.25
        
        stage_2 = (is_moving_to_target & (is_quat_aligned | is_rotating_to) &
                   is_gripper_opening & is_above).float() * 0.5

        stage_3 = (is_reached & is_gripper_closing).float() * 0.75

        stage_4 = (is_grasped).float() * 1.0

        reward = torch.stack([stage_1, stage_2, stage_3, stage_4], dim=0).max(dim=0).values


        if is_debug:
            print(is_grasped)
            print("-----------------")
        return reward

        