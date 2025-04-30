from copy import deepcopy
import os
import random
import time
from typing import Any, Dict, Optional, Tuple
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import transforms3d.euler as euler
from transforms3d import affines
from transforms3d import quaternions
from grasp_env_utils import (
    default_mapping,
    get_final_bodies_pose,
    get_key_bodies_pose,
    set_position_kinematics,
    transform_wirst_pos_to_obj,
)
from load_complex_obj import add_graspable_body, add_meshes_from_folder
from numpy.typing import NDArray
from dataclasses import dataclass, field
from pytorch_icem import iCEM, icem
from joblib import Parallel, delayed
from torch import Tensor
import torch


from gymnasium import error, spaces

from reach_pose_env import ReachPoseEnv, ReachPoseEnvConfig, convert_pose_dexgraspnet_to_mujoco


def run_mpc():
    POSE_NUM = 10

    obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
    pos_path_name = "final_positions/" + obj_name + ".npy"
    mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"
    model_path_hand = "./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"

    obj_quat_good = euler.euler2quat(np.deg2rad(90), np.deg2rad(180), np.deg2rad(180))
    obj_pos_good = [0.0, 0, 0.4]

    obj_start_pos = [0.0, 0, -0.5]

    obj_start_quat = euler.euler2quat(np.deg2rad(20), np.deg2rad(90), np.deg2rad(180))

    core_mug = np.load(pos_path_name, allow_pickle=True)
    qpos_hand = core_mug[POSE_NUM]["qpos"]

    wirst_pos = np.array(
        [
            qpos_hand["WRJTx"],
            qpos_hand["WRJTy"],
            qpos_hand["WRJTz"],
        ]
    )
    wirst_quat = euler.euler2quat(
        qpos_hand["WRJRx"],
        qpos_hand["WRJRy"],
        qpos_hand["WRJRz"],
    )

    transformed_wirst_pose, transformed_rot = transform_wirst_pos_to_obj(
        obj_start_pos, obj_start_quat, wirst_pos, wirst_quat
    )

    transformed_wirst_euler_ang = euler.mat2euler(transformed_rot)

    new_wirst_pos = {
        "WRJRx": transformed_wirst_euler_ang[0],
        "WRJRy": transformed_wirst_euler_ang[1],
        "WRJRz": transformed_wirst_euler_ang[2],
        "WRJTx": transformed_wirst_pose[0],
        "WRJTy": transformed_wirst_pose[1],
        "WRJTz": transformed_wirst_pose[2],
    }

    for key in new_wirst_pos.keys():
        qpos_hand[key] = new_wirst_pos[key]

    final_act_pose_sh_hand = convert_pose_dexgraspnet_to_mujoco(qpos_hand, default_mapping)
    key_body_final_pos = get_final_bodies_pose(qpos_hand, model_path_hand)

    config = ReachPoseEnvConfig(
        hand_final_full_pose=qpos_hand,
        model_path_hand=model_path_hand,
        obj_mesh_path=mesh_path,
        obj_start_pos=obj_start_pos,
        obj_start_quat=obj_start_quat,
        obj_scale=core_mug[POSE_NUM]["scale"] * 0.9,
        frame_skip=4,
    )

    reacher = ReachPoseEnv(config, key_pose_dict=key_body_final_pos)
    reacher_dynamic = ReachPoseEnv(config, key_pose_dict=key_body_final_pos)
    obs_mpc_state = reacher.reset_mpc()
    nu = reacher.action_space.shape[0]

    # create controller with chosen parameters
    def dynamics_mpc_wrapper(state_vec: Tensor, action_vec: Tensor):
        return reacher_dynamic.parralel_step_mpc(state_vec, action_vec)

    def cost_vec(sate_vec, action_vec):
        costs = []
        for trj in sate_vec:
            accum_cost_trj = 0
            for state, act_horizon in zip(trj, action_vec):
                step_cost, _ = reacher_dynamic.cost_mpc(state, act_horizon[0])
                accum_cost_trj += step_cost
            costs.append(accum_cost_trj)
            costs_tensor = Tensor(costs)
        return costs_tensor

    initial_mean = (reacher.action_space.low + reacher.action_space.high) / 2
    initial_sigma = reacher.action_space.high - initial_mean

    ctrl = iCEM(
        dynamics=dynamics_mpc_wrapper,
        trajectory_cost=cost_vec,
        sigma=Tensor(initial_sigma),
        nx=68,
        nu=nu,
        warmup_iters=20,
        online_iters=20,
        num_samples=100,
        num_elites=10,
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
        obs_mpc_state = reacher.reset_mpc()

        for i in range(MPC_STEPS):
            action_i = ctrl.command(obs_mpc_state, shift_nominal_trajectory=False)
            action_np = action_i.cpu().numpy()
            obs_mpc_state = reacher.step_mpc(obs_mpc_state, action_np)

            cost, decopose = reacher.cost_mpc(obs_mpc_state, action_np)
            costs_seq.append(cost)
            action_seq.append(action_np)

            print(f"Cost : {cost}")
            # print(f"Action : {action_np}")
            print(f"Step : {i}")
            mean_normilize = ctrl.mean[0]
            std_normilize = ctrl.std[0] / initial_sigma
            print(f"MEAN[0] {np.array(mean_normilize).round(3)}")
            print()
            print(f"STD[0] {np.array(std_normilize).round(3)}")

            # ctrl.shift()
        reacher.close()
        print("Finish traj generate")
        np.savez(obj_name + str(POSE_NUM), action_seq)
        np.savez(obj_name + str("_cost_") + str(POSE_NUM), costs_seq)
    else:
        print("File traj already exists")
        action_seq = np.load(obj_name + str(POSE_NUM) + ".npz")["arr_0"]
    obs_mpc_state = reacher.reset_mpc()
    # reacher.kinematics_debug = True

    viewer = mujoco.viewer.launch_passive(reacher.model, reacher.data)
    dist_reward = []
    obj_speed_reward = []
    wirst_orint_reward = []

    for action_i in action_seq:
        reacher.data.ctrl = action_i
        for i in range(reacher.frame_skip):
            mujoco.mj_step(reacher.model, reacher.data)
            for i in range(50):
                time.sleep(0.005)
                viewer.sync()
                joint_id_x = reacher.data.model.joint(name="obj_t_joint_x").id
                joint_id_y = reacher.data.model.joint(name="obj_t_joint_y").id
                joint_id_z = reacher.data.model.joint(name="obj_t_joint_z").id
                obj_speed_x = reacher.data.qvel[joint_id_x]
                obj_speed_y = reacher.data.qvel[joint_id_y]
                obj_speed_z = reacher.data.qvel[joint_id_z]
                obj_speed = np.linalg.norm(np.array([obj_speed_x, obj_speed_y, obj_speed_z]))

            full_obs = reacher._get_full_obs()
            distance_key_points_array = full_obs["distance_key_points_array"]
            obj_speed = full_obs["obj_speed"]
            anglediff = full_obs["anglediff"]
            obj_pose_err = full_obs["object_error"]
            rew, decompose = reacher.reward(distance_key_points_array, obj_pose_err, anglediff)
            dist_reward.append(decompose[0])
            obj_speed_reward.append(decompose[1])
            wirst_orint_reward.append(decompose[2])
    reacher.kinematics_debug = True
    for i in range(100):
        time.sleep(0.01)
        viewer.sync()

        if reacher.kinematics_debug:
            reacher.step(0)
        else:
            mujoco.mj_step(reacher.model, reacher.data)
        joint_id_x = reacher.data.model.joint(name="obj_t_joint_x").id
        joint_id_y = reacher.data.model.joint(name="obj_t_joint_y").id
        joint_id_z = reacher.data.model.joint(name="obj_t_joint_z").id
        obj_speed_x = reacher.data.qvel[joint_id_x]
        obj_speed_y = reacher.data.qvel[joint_id_y]
        obj_speed_z = reacher.data.qvel[joint_id_z]
        obj_speed = np.linalg.norm(np.array([obj_speed_x, obj_speed_y, obj_speed_z]))
        print(f"Object speed: {obj_speed}")

    reacher.reset_mpc()

    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(dist_reward, label="r1")
    plt.xlabel("Time step")
    plt.ylabel("Distance reward")

    plt.figure()

    plt.plot(obj_speed_reward, label="r2")
    plt.xlabel("Time step")
    plt.ylabel("Obj_speed reward")

    plt.figure()

    plt.plot(wirst_orint_reward, label="r3")
    plt.xlabel("Time step")
    plt.ylabel("wirst_orint_reward")
    plt.show()


if __name__ == "__main__":
    run_mpc()
