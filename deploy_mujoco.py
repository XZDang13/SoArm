import os
os.environ["MUJOCO_GL"] = "egl"

import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
from model.actor_critic import EncoderNet, StochasticDDPGActor
from RLAlg.nn.steps import DeterministicContinuousPolicyStep

device = torch.device("cuda:0")
encoder = EncoderNet(6+6+6+3+4, [256, 256, 256]).to(device)
actor = StochasticDDPGActor(encoder.dim, [256, 256], 6).to(device)

encoder_params, actor_params, _ = torch.load("model.pth")
encoder.load_state_dict(encoder_params)
actor.load_state_dict(actor_params)

encoder.eval()
actor.eval()

def get_body_pose(m, d, body_name: str):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    # World-frame position and orientation (quat = [w, x, y, z])
    pos  = d.xpos[bid].copy()
    quat = d.xquat[bid].copy()
    return pos, quat

def get_joint_qpos(m, d, joint_name: str):
    jid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr  = m.jnt_qposadr[jid]
    jtyp = m.jnt_type[jid]

    if jtyp == mujoco.mjtJoint.mjJNT_HINGE:   # 1 DoF angle (rad)
        return d.qpos[adr].copy()
    if jtyp == mujoco.mjtJoint.mjJNT_SLIDER:  # 1 DoF translation (m)
        return d.qpos[adr].copy()
    if jtyp == mujoco.mjtJoint.mjJNT_BALL:    # 4 values (quat w,x,y,z)
        return d.qpos[adr:adr+4].copy()
    if jtyp == mujoco.mjtJoint.mjJNT_FREE:    # 7 values: xyz(3) + quat(4)
        q = d.qpos[adr:adr+7].copy()
        pos, quat = q[:3], q[3:]             # quat = [w, x, y, z]
        return pos, quat

    raise ValueError("Unknown joint type")

def get_joints_qpos(m, d):

    robot_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    robot_qpos = []
    for name in robot_joint_names:
        val = get_joint_qpos(m, d, name)  # returns scalar for hinge/slider
        robot_qpos.append(float(val))
    robot_qpos = np.array(robot_qpos)

    return robot_qpos

def sample_cube_xy_yaw_np(n=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x   = rng.uniform(-0.10,  -0.10,  size=n)
    y   = rng.uniform(-0.175, -0.175, size=n)
    yaw = rng.uniform(-np.pi/4, np.pi/4, size=n)
    return x, y, yaw

# yaw (about Z) -> MuJoCo quat [w, x, y, z]
def yaw_to_quat_np(yaw):
    half = yaw / 2.0
    qw = np.cos(half)
    qx = np.zeros_like(yaw)
    qy = np.zeros_like(yaw)
    qz = np.sin(half)
    return np.stack([qw, qx, qy, qz], axis=-1)

def set_cube_pose_random_np(m, d, cube_free_joint="cube_free", cube_geom="cube_geom",
                            rng=None, margin=1e-3):
    # sample one pose
    x, y, yaw = sample_cube_xy_yaw_np(1, rng=rng)
    quat = yaw_to_quat_np(yaw)[0]  # shape (4,)

    # put cube on the ground: z = half-height + margin
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, cube_geom)
    half_z = m.geom_size[gid][2]   # for box geom: size = half extents
    z = float(half_z + margin)

    # write into the FREE joint's qpos slice (xyz + quat[w,x,y,z])
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cube_free_joint)
    adr = m.jnt_qposadr[jid]
    d.qpos[adr:adr+7] = np.array([x[0], y[0], z, *quat], dtype=float)

    mujoco.mj_forward(m, d)

@torch.no_grad()
def get_action(obs):
    obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)

    feature = encoder(obs)
    step:DeterministicContinuousPolicyStep = actor(feature, std=1.0)
    action = step.mean.squeeze(0).cpu().numpy()

    return action

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

m = mujoco.MjModel.from_xml_path("env/assets/so101/scene.xml")
d = mujoco.MjData(m)
m.opt.timestep = 1/30

set_cube_pose_random_np(m, d, cube_free_joint="cube_free", cube_geom="cube_geom")

goal_state = np.array([ 2.0074e-01, -1.6178e-01,  1.6200e-02,  9.6920e-01, -5.9605e-08, 1.4901e-08,  2.4629e-01])
pre_pos = get_joints_qpos(m, d)
current_pos = get_joints_qpos(m, d)
pre_action = np.array([0, 0, 0, 0, 0, 0])

print(m.jnt_range[:, 0])
print(m.jnt_range[:, 1])

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after simulation_duration wall-seconds.
    start = time.time()
    while viewer.is_running() and time.time() - start < 5:
        step_start = time.time()

        obs = np.concatenate([goal_state, current_pos, pre_pos, pre_action])
        action = get_action(obs)
        target_pos = current_pos + action * 0.25

        
        target_pos = target_pos.clip(m.jnt_range[:6, 0], m.jnt_range[:6, 1])

        
        d.qpos[:6] = target_pos
        #tau = pd_control(target_pos, d.qpos, 17.8, np.zeros_like(d.qvel), d.qvel, 0.6)
        #d.ctrl[:] = tau
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d)

        pre_pos = current_pos.copy()
        current_pos = get_joints_qpos(m, d)
        pre_action = action.copy()

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    viewer.close()
