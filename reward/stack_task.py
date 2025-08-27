import torch
from .motion_detector import MotionDetector

class StackTaskReward:
    @staticmethod
    def compute_reward(gripper_pos_from: torch.Tensor,
                       gripper_pos_to: torch.Tensor,
                       target_pos: torch.Tensor) -> torch.Tensor:
        
        is_above = MotionDetector.is_above(gripper_pos_to, target_pos,
                                           aligned=True, xy_align_tol=0.01,
                                           z_band=0.1).float()

        is_moving_to = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.6,
            offset=torch.tensor([0.0, 0.0, 0.1], device=gripper_pos_from.device),
        ).float()

        is_moving_to *= 0.5
        is_above *= 1.0

        reward = torch.max(is_moving_to, is_above)

        return reward

        