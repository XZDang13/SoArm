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
                       is_debug:bool) -> torch.Tensor:
        
        is_moving = MotionDetector.is_moving(gripper_pos_from, gripper_pos_to, threshold=0.005)
        
        is_above = MotionDetector.is_above(gripper_pos_to, target_pos,
                                           aligned=True, xy_align_tol=0.05, z_band=0.2)

        is_reached = MotionDetector.is_reached(
            gripper_pos_to, target_pos, 0.01
        )

        is_moving_to = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.5
        )

        is_moving_to_top = MotionDetector.is_moving_to(
            gripper_pos_from, gripper_pos_to,
            target_pos, threshold=0.5,
            offset=torch.tensor([0.0, 0.0, 0.15], device=gripper_pos_from.device),
        )

        is_rotating_to = MotionDetector.is_rotating_to_quat(
            gripper_quat_from, gripper_quat_to,
            target_quat, tol_decrease=0.01
        )

        is_quat_aligned = MotionDetector.is_quat_reached(
            gripper_quat_to, target_quat, threshold_rad=(25 * torch.pi/180)
        )

        is_gripper_opening = (MotionDetector.is_gripper_opened(gripper_joint_pos_to, torch.pi/4) |
                              MotionDetector.is_openning_joint_pos(gripper_joint_pos_from, gripper_joint_pos_to))
        
        is_gripper_closing = MotionDetector.is_closing_joint_pos(gripper_joint_pos_from, gripper_joint_pos_to)

        stage_1 = (is_moving_to_top & is_rotating_to & is_gripper_opening & is_moving & (~is_above)).float() * 0.25

        stage_2 = (is_moving_to & is_above &
                (is_quat_aligned | (is_rotating_to & is_moving)) &
                is_gripper_opening).float() * 0.5

        stage_3 = (is_reached & is_quat_aligned & is_gripper_closing).float() * 0.75

        stage_4 = (is_grasped & (~is_moving)).float() * 1.0

        reward = torch.stack([stage_1, stage_2, stage_3, stage_4], dim=0).max(dim=0).values

        if is_debug:
            print(reward)
            #print(is_reahched)
            #print(is_graspped)
            #print(is_quat_aligned)
            #print(is_moving_to)
            #print(is_above)
            #print(is_quat_aligned)
            print("-----------------")

        return reward

        