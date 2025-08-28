import torch
from .motion_detector import MotionDetector

class ReachTaskReward:
    @staticmethod
    def compute_reward(gripper_pos_from: torch.Tensor,
                       gripper_quat_from: torch.Tensor,
                       gripper_pos_to: torch.Tensor,
                       gripper_quat_to: torch.Tensor,
                       target_pos: torch.Tensor,
                       target_quat: torch.Tensor) -> torch.Tensor:
        
        is_reahched = MotionDetector.is_reached(
            gripper_pos_to, target_pos, 0.015
        )

        is_moving_to = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.6
        )

        is_rotating_to = MotionDetector.is_rotating_to_quat(
            gripper_quat_from, gripper_quat_to,
            target_quat, tol_decrease=0.01
        )

        is_quat_aligned = MotionDetector.is_quat_reached(
            gripper_quat_to, target_quat
        )

        stage_1 = (is_moving_to & is_rotating_to).float() * 0.5
        stage_2 = (is_reahched & is_quat_aligned).float() * 1.0
        
        reward = torch.stack([stage_1, stage_2], dim=0).max(dim=0).values

        return reward

        