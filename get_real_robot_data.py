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

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

LOWER_LIMITS = np.array([-1.91986218, -1.74532925, -1.69, -1.65806285, -2.7438473,  -0.17453298])
UPPER_LIMITS = np.array([1.91986218, 1.74532925, 1.69, 1.65806273, 2.84120631, 1.7453292])

@dataclass
class SetupConfig:
    robot: RobotConfig
    play_sounds: bool = True

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

def get_camera_obs(obs):
    img = obs["camera"]

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
        port="/dev/ttyACM0", id="my_follower_arm", use_degrees=True
    )

    robot = SO101Follower(robot_config)
    robot.cameras = {"camera": camera}
    print("init")
    robot.connect()
    count = 0

    win_name = "OpenCV Camera (LeRobot)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        log_say("Start", True, blocking=True)
        init_state = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': 0.0,
            'elbow_flex.pos': 0.0,
            'wrist_flex.pos': 90.0,
            'wrist_roll.pos': 0.0,
            'gripper.pos': 0.0
        }

        move_to_state(robot, init_state)

        while True:
            obs = robot.get_observation()
            frame = get_camera_obs(obs)
            joint_pos = get_joint_pos(obs)

            color_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(win_name, color_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("ESC pressed, exiting.")
                break
            elif key == ord("s"):  # Save frame
                print(joint_pos)
                print(np.deg2rad(joint_pos))
                filename = f"debug_data/real_{count}.png"
                cv2.imwrite(filename, color_bgr)
                count += 1

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()