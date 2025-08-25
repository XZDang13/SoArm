import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
from model.actor_critic import EncoderNet, StochasticDDPGActor
from RLAlg.nn.steps import DeterministicContinuousPolicyStep

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

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

m = mujoco.MjModel.from_xml_path("env/assets/so101/scene.xml")
d = mujoco.MjData(m)
m.opt.timestep = 1/60

goal_state = np.array([0.25, 0.0, 0.17, 1.0, 0.0, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0])
pre_pos = d.qpos[:].copy()
current_pos = d.qpos[:].copy()
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

        
        target_pos = target_pos.clip(m.jnt_range[:, 0], m.jnt_range[:, 1])

        for _ in range(2):
            #d.qpos[:] = target_pos
            tau = pd_control(target_pos, d.qpos, 17.8, np.zeros_like(d.qvel), d.qvel, 0.6)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

        pre_pos = current_pos.copy()
        current_pos = d.qpos[:].copy()
        pre_action = action.copy()

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    viewer.close()
