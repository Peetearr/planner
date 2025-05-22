import time
import mujoco
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt

from icem_mpc.reach_pose_env import ReachPoseEnv


def trajectory_player(reacher_pose_env: ReachPoseEnv, action_seq, flying_camera = True):
    viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    dist_reward = []
    obj_displacement_reward = []
    wirst_orint_reward = []
    
    viewer.cam.distance = 1
    viewer.cam.elevation = -20
    viewer.cam.lookat[2] += -0.25

    for action_i in action_seq:
        reacher_pose_env.data.ctrl = action_i
        for i in range(reacher_pose_env.frame_skip):
            mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
            for i in range(25):
                time.sleep(0.002)
                if flying_camera:
                    viewer.cam.elevation += 0.005
                viewer.sync()
            full_obs = reacher_pose_env._get_full_obs()
            rew, decompose = reacher_pose_env.reward(full_obs, action_i)
            dist_reward.append(decompose["distance_key_points"])
            obj_displacement_reward.append(decompose["obj_displacement"])
            wirst_orint_reward.append(decompose["diff_orient"])

        viewer.cam.trackbodyid = reacher_pose_env.data.model.body(name="graspable_object").id


    reacher_pose_env.kinematics_debug = False
    for i in range(1000):
        time.sleep(0.01)
        viewer.sync()
        if reacher_pose_env.kinematics_debug:
            reacher_pose_env.render_mode = None
            reacher_pose_env.step(0)
        else:
            if flying_camera:
                viewer.cam.azimuth += 0.4
    viewer.close()


    # plt.figure()

    # plt.plot(dist_reward, label="r1")
    # plt.xlabel("Time step")
    # plt.ylabel("Distance reward")

    # plt.figure()

    # plt.plot(obj_displacement_reward, label="r2")
    # plt.xlabel("Time step")
    # plt.ylabel("Obj displacement reward")

    # plt.figure()

    # plt.plot(wirst_orint_reward, label="r3")
    # plt.xlabel("Time step")
    # plt.ylabel("wirst_orint_reward")
    # plt.show()


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
