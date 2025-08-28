import torch
from .motion_detector import MotionDetector, quat_geodesic_angle

class StackTaskReward:
    @staticmethod
    def compute_reward(gripper_pos_from: torch.Tensor,
                       gripper_quat_from: torch.Tensor,
                       gripper_pos_to: torch.Tensor,
                       gripper_quat_to: torch.Tensor,
                       target_pos: torch.Tensor,
                       target_quat: torch.Tensor) -> torch.Tensor:
        
        is_above = MotionDetector.is_above(gripper_pos_to, target_pos,
                                           aligned=True, xy_align_tol=0.05, z_band=0.2)

        is_reahched = MotionDetector.is_reached(
            gripper_pos_to, target_pos, 0.015
        )

        is_moving_to = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.5
        )

        is_moving_to_top = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.5,
            offset=torch.tensor([0.0, 0.0, 0.125], device=gripper_pos_from.device),
        )

        is_rotating_to = MotionDetector.is_rotating_to_quat(
            gripper_quat_from, gripper_quat_to,
            target_quat, tol_decrease=0.01
        )

        is_quat_aligned = MotionDetector.is_quat_reached(
            gripper_quat_to, target_quat, threshold_rad=(25 * torch.pi/180)
        )

        stage_1 = (is_moving_to_top & is_rotating_to).float() * 0.33
        stage_2 = (is_moving_to & is_above & (is_quat_aligned | is_rotating_to)).float() * 0.666
        stage_3 = (is_reahched & is_quat_aligned).float() * 1.0

        reward = torch.stack([stage_1, stage_2, stage_3], dim=0).max(dim=0).values

        #print(reward)
        #print(is_reahched)
        #print(is_quat_aligned)
        #print(is_moving_to)
        #print(is_above)
        #print(is_quat_aligned)
        #print("-----------------")

        return reward

        