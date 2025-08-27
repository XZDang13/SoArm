import torch
from .motion_detector import MotionDetector

class StackTaskReward:
    @staticmethod
    def compute_reward(gripper_pos_from: torch.Tensor,
                       gripper_pos_to: torch.Tensor,
                       target_pos: torch.Tensor) -> torch.Tensor:
        
        is_above = MotionDetector.is_above(gripper_pos_to, target_pos,
                                           aligned=True, xy_align_tol=0.01)

        is_reahched = MotionDetector.is_reached(
            gripper_pos_to, target_pos, 0.025
        )

        is_moving_to = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.6
        )

        is_moving_to_top = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.6,
            offset=torch.tensor([0.0, 0.0, 0.1], device=gripper_pos_from.device),
        )

        stage_1 = (is_moving_to_top).float() * 0.333
        stage_2 = (is_moving_to & is_above).float() * 0.666
        stage_3 = (is_reahched).float() * 1.0

        reward = torch.stack([stage_1, stage_2, stage_3], dim=0).max(dim=0).values

        return reward

        