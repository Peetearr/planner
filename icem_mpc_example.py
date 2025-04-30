import os
import time
import mujoco
import numpy as np
from config_crearor import prepare_env_config
from pytorch_icem import iCEM
from torch import Tensor

from reach_pose_env import ReachPoseEnv
import gymnasium as gym
from functools import partial
import matplotlib.pyplot as plt

def trajectory_player(reacher_pose_env: ReachPoseEnv, action_seq):
    viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    dist_reward = []
    obj_displacement_reward = []
    wirst_orint_reward = []

    for action_i in action_seq:
        reacher_pose_env.data.ctrl = action_i
        for i in range(reacher_pose_env.frame_skip):
            mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
            for i in range(50):
                time.sleep(0.005)
                viewer.sync()
            full_obs = reacher_pose_env._get_full_obs()
            rew, decompose = reacher_pose_env.reward(full_obs, action_i)
            dist_reward.append(decompose["distance_key_points"])
            obj_displacement_reward.append(decompose["obj_displacement"])
            wirst_orint_reward.append(decompose["diff_orient"])

    reacher_pose_env.kinematics_debug = True
    for i in range(100):
        time.sleep(0.01)
        viewer.sync()
        if reacher_pose_env.kinematics_debug:
            reacher_pose_env.step(0)
        else:
            mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
            
    plt.figure()

    plt.plot(dist_reward, label="r1")
    plt.xlabel("Time step")
    plt.ylabel("Distance reward")

    plt.figure()

    plt.plot(obj_displacement_reward, label="r2")
    plt.xlabel("Time step")
    plt.ylabel("Obj displacement reward")

    plt.figure()

    plt.plot(wirst_orint_reward, label="r3")
    plt.xlabel("Time step")
    plt.ylabel("wirst_orint_reward")
    plt.show()


def dynamics_mpc_wrapper_np(env: ReachPoseEnv, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    res_states = np.zeros(state_vec.shape, dtype=state_vec.dtype)
    for state, act, i in zip(state_vec, action_vec, range(state_vec.shape[0])):
        env.set_state(state)
        reduced_obs, reward, _, _, debug_dict = env.step(act)
        state = debug_dict["state"]
        res_states[i, :] = state
    return res_states


def dynamics_mpc_wrapper_tensor(env: ReachPoseEnv, state_vec: Tensor, action_vec: Tensor):
    res_states = dynamics_mpc_wrapper_np(env, state_vec.cpu().numpy(), action_vec.cpu().numpy())
    res_states_tensor = Tensor(res_states)
    return res_states_tensor


def cost_traj_mpc_np(env: ReachPoseEnv, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    costs = np.zeros(state_vec.shape[0], dtype=np.float32)
    for trj, act_horizon, i in zip(state_vec, action_vec, range(state_vec.shape[0])):
        accum_cost_trj = 0
        for state_i, act_i in zip(trj, act_horizon):
            env.set_state(state_i)
            obs = env._get_full_obs()
            step_cost, _ = env.reward(obs, act_i)
            accum_cost_trj += -step_cost
        costs[i] = accum_cost_trj

    return costs


def cost_traj_mpc_tensor(env: ReachPoseEnv, state_vec: Tensor, action_vec: Tensor) -> Tensor:
    costs = cost_traj_mpc_np(env, state_vec.cpu().numpy(), action_vec.cpu().numpy())
    costs_tensor = Tensor(costs)
    return costs_tensor


def run_mpc():
    POSE_NUM, obj_name, key_body_final_pos, config, final_act_pose_sh_hand = prepare_env_config()

    reacher = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human")
    reacher_dynamic = ReachPoseEnv(config, key_pose_dict=key_body_final_pos)

    vector_dynamics = partial(dynamics_mpc_wrapper_tensor, reacher_dynamic)
    vector_cost = partial(cost_traj_mpc_tensor, reacher_dynamic)

    _, info = reacher.reset()
    start_state = reacher.get_state()
    nu = reacher.action_space.shape[0]

    initial_mean = (reacher.action_space.low + reacher.action_space.high) / 2
    initial_sigma = reacher.action_space.high - initial_mean

    ctrl = iCEM(
        dynamics=vector_dynamics,
        trajectory_cost=vector_cost,
        sigma=Tensor(initial_sigma),
        nx=start_state.shape[0],
        nu=nu,
        warmup_iters=20,
        online_iters=20,
        num_samples=150,
        num_elites=20,
        elites_keep_fraction=0.5,
        horizon=7,
        device="cpu",
        alpha=0.005,
        noise_beta=2,
        low_bound_action=reacher.action_space.low,
        high_bound_action=reacher.action_space.high,
    )

    ctrl.mean = Tensor(initial_mean)
    ctrl.sigma = Tensor(initial_sigma)

    if not os.path.exists(obj_name + str(POSE_NUM) + ".npz"):

        MPC_STEPS = 40
        action_seq = []
        costs_seq = []
        obs_mpc_state = start_state

        for i in range(MPC_STEPS):
            action_i = ctrl.command(obs_mpc_state, shift_nominal_trajectory=False)
            action_np = action_i.cpu().numpy()
            reduced_obs, reward, _, _, debug_dict = reacher.step(action_np)
            obs_mpc_state = reacher.get_state()

            costs_seq.append(reward)
            action_seq.append(action_np)

            print(f"Cost : {reward}")
            print(f"Step : {i}")
            mean_normilize = ctrl.mean[0]
            std_normilize = ctrl.std[0] / initial_sigma
            print(f"MEAN[0] {np.array(mean_normilize).round(3)}")
            print()
            print(f"STD[0] {np.array(std_normilize).round(3)}")

            ctrl.shift()
        reacher.close()
        print("Finish traj generate")
        np.savez(obj_name + str(POSE_NUM), action_seq)
        np.savez(obj_name + str("_cost_") + str(POSE_NUM), costs_seq)
    else:
        print("File traj already exists")
        action_seq = np.load(obj_name + str(POSE_NUM) + ".npz")["arr_0"]
    obs_mpc_state = reacher.reset()


    trajectory_player(reacher, action_seq)




if __name__ == "__main__":
    run_mpc()
