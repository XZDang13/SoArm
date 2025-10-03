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
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)


import cv2
import numpy as np
from PIL import Image


LOWER_LIMITS = np.array([-1.91986218, -1.74532925, -1.69, -1.65806285, -2.7438473,  -0.17453298])
UPPER_LIMITS = np.array([1.91986218, 1.74532925, 1.69, 1.65806273, 2.84120631, 1.7453292])

logger = logging.getLogger(__name__)

@dataclass
class SetupConfig:
    robot: RobotConfig
    play_sounds: bool = True


deg_range = {
    "shoulder_pan": (-110, 110),
    "shoulder_lift": (-100, 100),
    "elbow_flex": (-96, 96),
    "wrist_flex": (-95, 95),
    "wrist_roll": (-157, 162),
    "gripper": (-10, 100)
}

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
    robot.connect(calibrate=False)
    count = 0

    try:
        if robot.calibration:
            # self.calibration is not empty here
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {robot.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {robot.id} to the motors")
                robot.bus.write_calibration(robot.calibration)
                return

        logger.info(f"\nRunning calibration of {robot}")
        robot.bus.disable_torque()
        for motor in robot.bus.motors:
            robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {robot} to the middle of its range of motion and press ENTER....")
        homing_offsets = robot.bus.set_half_turn_homings()
        print(homing_offsets)
        
        range_mins = {}
        range_maxes = {}
        for motor in robot.bus.motors.keys():
            deg_min, deg_max = deg_range[motor]
            range_min = deg_min * 4096 / 360 + 2047
            range_max = deg_max * 4096 / 360 + 2047
            range_mins[motor] = int(range_min)
            range_maxes[motor] = int(range_max)
            print(f"{motor}: {range_mins[motor]}, {range_maxes[motor]}")

        robot.calibration = {}
        for motor, m in robot.bus.motors.items():
            robot.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        robot.bus.write_calibration(robot.calibration)
        robot._save_calibration()
        print("Calibration saved to", robot.calibration_fpath)
            

    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()