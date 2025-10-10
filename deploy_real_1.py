import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.utils import log_say
from lerobot.cameras.configs import ColorMode, Cv2Rotation

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
from model.actor_critic import MobileFrameObservationEncoderNet, StochasticDDPGActor, FrameObservationEncoderNet
from RLAlg.nn.steps import DeterministicContinuousPolicyStep


LOWER_LIMITS = np.array([-1.91986218, -1.74532925, -1.69, -1.65806285, -2.7438473,  -0.17453298])
UPPER_LIMITS = np.array([1.91986218, 1.74532925, 1.69, 1.65806273, 2.84120631, 1.7453292])


def get_camera_obs(obs):
    camera = obs["camera"]
    img = Image.fromarray(camera)
    #alpha = 0.
    #img = Image.blend(img, sim_backgound, alpha).convert("RGB")

    return img

def get_joint_pos(obs):
    joint_pos = np.array([
        obs['shoulder_pan.pos'],
        obs['shoulder_lift.pos'],
        obs['elbow_flex.pos'],
        obs['wrist_flex.pos'],
        obs['wrist_roll.pos'],
        obs['gripper.pos']
    ])
    return joint_pos

def get_cmd(target_pos):
    cmd = {
        'shoulder_pan.pos': target_pos[0],
        'shoulder_lift.pos': target_pos[1],
        'elbow_flex.pos': target_pos[2],
        'wrist_flex.pos': target_pos[3],
        'wrist_roll.pos': target_pos[4],
        'gripper.pos': target_pos[5]
    }
    return cmd

def move_to_state(robot:Robot, state:dict):
    duration = 2.0   # seconds
    rate = 30.0      # Hz
    steps = int(duration * rate)

    robot_state = robot.get_observation()
    current_state = {name: robot_state[name] for name in state.keys()}

    trajectory = []
    for t in range(steps + 1):
        alpha = t / steps  # goes from 0 → 1
        action = {}
        for joint in state:
            start = current_state[joint]
            target = state[joint]
            action[joint] = (1 - alpha) * start + alpha * target
            
        trajectory.append(action)

    for action in trajectory:
        loop_start = time.perf_counter()
        
        robot.send_action(action)
        
        dt_s = time.perf_counter() - loop_start
        sleep_time = 1.0 / rate - dt_s
        busy_wait(sleep_time)

class PolicyController:
    def __init__(self):
        self.device = torch.device("cuda:0")

        self.frame_encoder = MobileFrameObservationEncoderNet(128).to(self.device)
        self.actor = StochasticDDPGActor(256, [256, 256], 6).to(self.device)

        frame_encoder_params, actor_params, _ = torch.load("frame_model.pth")
        self.frame_encoder.load_state_dict(frame_encoder_params)
        self.actor.load_state_dict(actor_params)
        self.frame_encoder.eval()
        self.actor.eval()


        self._action_scale = 0.05

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def reset(self, frame):
        self.pre_frame = frame
        self.count = 0

    def compute_features(self, frame):
        pre_frame = self.pre_frame.copy()
        
        frames = [frame, pre_frame]

        frames = torch.stack(self.transform(frames), dim=0).to(self.device)

        features = self.frame_encoder(frames, True)
        features = features.view(-1, 256)

        self.pre_frame = frame.copy()

        frame.save(f"imgs/{self.count}.png")
        self.count += 1

        return features
    
    @torch.no_grad()
    def get_action(self, frame, joint_pos):
        joint_pos = torch.from_numpy(joint_pos).to(self.device)
        joint_pos = torch.deg2rad(joint_pos)
        features = self.compute_features(frame)
        step = self.actor(features, std=1.0)    
        action = step.mean.squeeze(0)

        target_joint_pos = action * self._action_scale + joint_pos
        print(action)
        return target_joint_pos.cpu().numpy()


def main():
    camera = OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION
        )
    )

    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0", id="my_follower_arm"
    )

    robot = SO101Follower(robot_config)
    robot.cameras = {"camera": camera}
    controller = PolicyController()
    print("init")
    robot.connect()
    count = 0
    try:
        log_say("Settin to init state", True, blocking=True)
        init_state = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': 0.0,
            'elbow_flex.pos': 0.0,
            'wrist_flex.pos': 90.0,
            'wrist_roll.pos': 0.0,
            'gripper.pos': 0.0
        }
        move_to_state(robot, init_state)
        
        log_say("Inference", True, blocking=True)
        obs = robot.get_observation()
        frame = get_camera_obs(obs)
        controller.reset(frame)
        while True:
            loop_start = time.perf_counter()
            
            obs = robot.get_observation()
            frame = get_camera_obs(obs)
            joint_pos = get_joint_pos(obs)

            target_pos_rad = controller.get_action(frame, joint_pos)
            target_pos_rad = target_pos_rad.clip(LOWER_LIMITS, UPPER_LIMITS)
            target_pos = np.rad2deg(target_pos_rad).tolist()

            cmd = get_cmd(target_pos)
            #print(joint_pos)
            #print(target_pos)
            #print("------------")
            #print(action)
            #print(cmd)
            #print("-----------------")

            robot.send_action(cmd)

            dt_s = time.perf_counter() - loop_start
            sleep_time = 1.0 / 30 - dt_s
            busy_wait(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        rest_state = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': -100.0,
            'elbow_flex.pos': 100.0,
            'wrist_flex.pos': 65.0,
            'wrist_roll.pos': 0.0,
            'gripper.pos': 0.0
        }
        move_to_state(robot, rest_state)
        robot.disconnect()
    

if __name__ == "__main__":
    main()