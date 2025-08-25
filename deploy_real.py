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
import numpy as np
import torch
from model.actor_critic import EncoderNet, StochasticDDPGActor
from RLAlg.nn.steps import DeterministicContinuousPolicyStep


LOWER_LIMITS = np.array([-1.91986218, -1.74532925, -1.69, -1.65806285, -2.7438473,  -0.17453298])
UPPER_LIMITS = np.array([1.91986218, 1.74532925, 1.69, 1.65806273, 2.84120631, 1.7453292])

@dataclass
class SetupConfig:
    robot: RobotConfig
    play_sounds: bool = True

def get_joint_pos(robot:Robot):
    state = robot.get_observation()
    joint_pos = np.array([
        state['shoulder_pan.pos'],
        state['shoulder_lift.pos'],
        state['elbow_flex.pos'],
        state['wrist_flex.pos'],
        state['wrist_roll.pos'],
        state['gripper.pos']
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
            print(action)
        trajectory.append(action)

    for action in trajectory:
        loop_start = time.perf_counter()
        
        robot.send_action(action)
        
        dt_s = time.perf_counter() - loop_start
        sleep_time = 1.0 / rate - dt_s
        busy_wait(sleep_time)



@draccus.wrap()
def inference(cfg: SetupConfig):
    device = torch.device("cuda:0")
    encoder = EncoderNet(6+6+6+3+4+4, [256, 256, 256]).to(device)
    actor = StochasticDDPGActor(encoder.dim, [256, 256], 6).to(device)

    encoder_params, actor_params, _ = torch.load("model.pth")
    encoder.load_state_dict(encoder_params)
    actor.load_state_dict(actor_params)

    encoder.eval()
    actor.eval()

    @torch.no_grad()
    def get_action(obs):
        obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)

        feature = encoder(obs)
        step:DeterministicContinuousPolicyStep = actor(feature, std=1.0)
        action = step.mean.squeeze(0).cpu().numpy()

        return action
    
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    log_say("Starting", cfg.play_sounds, blocking=True)

    start = time.perf_counter()

    state = robot.get_observation()
    print(state)

    current_pos = np.deg2rad(get_joint_pos(robot))
    pre_pos = np.deg2rad(get_joint_pos(robot))
    pre_action = np.array([0, 0, 0, 0, 0, 0])
    goal_state = np.array([0.25, 0.0, 0.17, 1.0, 0.0, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0])

    try:
        log_say("Settin to init state", cfg.play_sounds, blocking=True)
        init_state = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': 0.0,
            'elbow_flex.pos': 0.0,
            'wrist_flex.pos': 0.0,
            'wrist_roll.pos': 0.0,
            'gripper.pos': 0.0
        }
        move_to_state(robot, init_state)
        
        log_say("Inference", cfg.play_sounds, blocking=True)
        while True:
            loop_start = time.perf_counter()
            obs = np.concatenate([goal_state, current_pos, pre_pos, pre_action], axis=-1)
            action = get_action(obs)
            target_pos_rad = current_pos + action * 0.25
            target_pos_rad = target_pos_rad.clip(LOWER_LIMITS, UPPER_LIMITS)
            target_pos = np.rad2deg(target_pos_rad).tolist()

            cmd = get_cmd(target_pos)

            print(target_pos_rad)
            print(action)
            print(cmd)
            print("-----------------")

            robot.send_action(cmd)
            dt_s = time.perf_counter() - loop_start
            sleep_time = 1.0 / 30 - dt_s
            busy_wait(sleep_time)
            pre_pos = current_pos.copy()
            current_pos = np.deg2rad(get_joint_pos(robot))
            pre_action = action.copy()

    except KeyboardInterrupt:
        pass
    finally:
        rest_state = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': -100.0,
            'elbow_flex.pos': 100.0,
            'wrist_flex.pos': 72.85,
            'wrist_roll.pos': 0.0,
            'gripper.pos': 0.0
        }
        move_to_state(robot, rest_state)
        robot.disconnect()

def main():
    inference()


if __name__ == "__main__":
    main()