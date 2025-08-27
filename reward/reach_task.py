import torch
from .motion_detector import MotionDetector

class ReachTaskReward:
    @staticmethod
    def compute_reward(gripper_pos_from: torch.Tensor,
                       gripper_pos_to: torch.Tensor,
                       target_pos: torch.Tensor) -> torch.Tensor:
        
        is_reahched = MotionDetector.is_reached(
            gripper_pos_to, target_pos, 0.025
        ).float()

        is_moving_to = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, 0.6
        ).float()

        is_moving_to *= 0.5
        is_reahched *= 1.0

        reward = torch.max(is_moving_to, is_reahched)

        return reward

        