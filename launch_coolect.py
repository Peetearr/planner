from dataclasses import dataclass
import os
import time
from typing import Optional
import mujoco
import numpy as np
from config_crearor import prepare_env_config
from pytorch_icem import iCEM
from torch import Tensor

from icem_mpc_example import trajectory_player, cost_traj_mpc_tensor, dynamics_mpc_wrapper_tensor
from reach_pose_env import ReachPoseEnv
import gymnasium as gym
from functools import partial
import matplotlib.pyplot as plt


@dataclass
class ConfigICEM:
    warmup_iters: int = 25
    online_iters: int = 25
    num_samples: int = 250
    num_elites: int = 20
    elites_keep_fraction: float = 0.5
    horizon: int = 7
    alpha: float = 0.005
    mpc_steps: int = 40


def run_icem(
    obj_name: str,
    pose_num: int,
    reward_dict: dict[str, float],
    confi_icem: ConfigICEM,
    frame_skip: int = 4,
    render_mode: Optional[str] = None,
):
    POSE_NUM, obj_name, key_body_final_pos, config, final_hand_joint_pose = prepare_env_config(
        obj_name=obj_name, pose_num=pose_num, frame_skip=frame_skip
    )
    reacher = ReachPoseEnv(
        config,
        key_pose_dict=key_body_final_pos,
        render_mode=render_mode,
        reward_dict=reward_dict,
    )
    reacher_dynamic = ReachPoseEnv(
        config, key_pose_dict=key_body_final_pos, reward_dict=reward_dict
    )

    vector_dynamics = partial(dynamics_mpc_wrapper_tensor, reacher_dynamic)
    vector_cost = partial(cost_traj_mpc_tensor, reacher_dynamic)

    initial_mean = (reacher.action_space.low + reacher.action_space.high) / 2
    initial_sigma = reacher.action_space.high - initial_mean

    start_state = reacher.get_state()
    nu = reacher.action_space.shape[0]

    ctrl = iCEM(
        dynamics=vector_dynamics,
        trajectory_cost=vector_cost,
        sigma=Tensor(initial_sigma),
        nx=start_state.shape[0],
        nu=nu,
        warmup_iters=confi_icem.warmup_iters,
        online_iters=confi_icem.online_iters,
        num_samples=confi_icem.num_samples,
        num_elites=confi_icem.num_elites,
        elites_keep_fraction=confi_icem.elites_keep_fraction,
        horizon=confi_icem.horizon,
        device="cpu",
        alpha=confi_icem.alpha,
        noise_beta=2,
        low_bound_action=reacher.action_space.low,
        high_bound_action=reacher.action_space.high,
    )

    ctrl.mean = Tensor(initial_mean)
    ctrl.sigma = Tensor(initial_sigma)

    action_seq = np.zeros((confi_icem.mpc_steps, nu), dtype=np.float32)
    costs_seq = np.zeros(confi_icem.mpc_steps)
    full_observations = []
    obs_mpc_state = start_state

    number_of_trj = int(ctrl.keep_fraction * ctrl.K)
    ellites_trj = np.zeros((number_of_trj, confi_icem.mpc_steps, nu), dtype=np.float32)

    for i in range(confi_icem.mpc_steps):
        action_i = ctrl.command(obs_mpc_state, shift_nominal_trajectory=False)
        action_np = action_i.cpu().numpy()
        reduced_obs, reward, _, _, debug_dict = reacher.step(action_np)
        obs_mpc_state = reacher.get_state()

        # Collect data
        costs_seq[i] = reward
        action_seq[i] = action_np
        ellites_trj[:, i, :] = ctrl.kept_elites[:, 0, :].to("cpu").numpy()
        full_observations.append(debug_dict["full_obs"])

        # Debugging
        if reacher.render_mode == "human":
            print(f"Cost : {reward}")
            print(f"Step : {i}")
            mean_normilize = ctrl.mean[0]
            std_normilize = ctrl.std[0] / initial_sigma
            print(f"MEAN[0] {np.array(mean_normilize).round(3)}")
            print()
            print(f"STD[0] {np.array(std_normilize).round(3)}")

        ctrl.shift()

    if reacher.render_mode == "human":
        reacher.close()

    return action_seq, costs_seq, ellites_trj, full_observations


if __name__ == "__main__":
    reward_dict = {
        "distance_key_points": 1.5,
        "obj_displacement": 10.0,
        "diff_orient": 3.0,
    }
    config_icem = ConfigICEM()
    obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
    pose_num = 5
                                                                                                                                                                                                        
    action_seq, costs_seq, ellites_trj, full_observations = run_icem(obj_name, pose_num, reward_dict, config_icem, render_mode="human")
