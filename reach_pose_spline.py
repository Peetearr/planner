from copy import deepcopy
import os
import random
import time
from typing import Dict, Optional, Tuple
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import transforms3d.euler as euler
from transforms3d import affines
from transforms3d import quaternions
from grasp_env_utils import (
    create_quintic_traj_function,
    default_mapping,
    get_final_bodies_pose,
    get_key_bodies_pose,
    quintic_func,
    set_position_kinematics,
    transform_wirst_pos_to_obj,
)
from load_complex_obj import add_graspable_body, add_meshes_from_folder
from numpy.typing import NDArray
from dataclasses import dataclass, field
from pytorch_icem import iCEM, icem
from pytorch_mppi import MPPI, SMPPI
from joblib import Parallel, delayed
from torch import Tensor
from reach_pose_env import ReachPoseEnv, ReachPoseEnvConfig, convert_pose_dexgraspnet_to_mujoco
import torch

def run_mpc():
    POSE_NUM = 3
    obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
    pos_path_name = "final_positions/" + obj_name + ".npy"
    mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"
    model_path_hand = "./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"


    obj_start_pos = np.array([0.0, -0.2, 0])
    #obj_start_pos = np.array([0.0, -0.1, 0])
    obj_start_quat = euler.euler2quat(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))

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
        obj_scale=core_mug[POSE_NUM]["scale"]*0.9,
        frame_skip = 3
    )

    reacher = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human")
    reacher_dynamic = ReachPoseEnv(config, key_pose_dict=key_body_final_pos)
    obs_mpc_state = reacher.reset_mpc()
 


    # create controller with chosen parameters
    def dynamics_mpc_wrapper(state_vec: Tensor, action_vec: Tensor):
        return reacher_dynamic.parralel_step_mpc(state_vec, action_vec)


    def runnng_cost(sate_vec, action_vec):
        costs = []
 
        for one_state in sate_vec:
            cost, _ = reacher_dynamic.cost_mpc(one_state, 0)
            costs.append(cost)
        costs_tensor = Tensor(costs)
        return costs_tensor

    high_action = Tensor(reacher.action_space.high)
    low_action = Tensor(reacher.action_space.low)
    wide_action = high_action - low_action
    noise_sigma = torch.diag(wide_action*0.5).double()
    nu = reacher.action_space.shape[0]
    MPC_STEPS = 100
    q0 = np.array([0, 0, 0])
    qf = np.array([1, 2, 3])
    traj_time =  MPC_STEPS*reacher.frame_skip*reacher.model.opt.timestep

    
    q_start = reacher.data.ctrl
    q_final = np.zeros(nu)

    for key, value in final_act_pose_sh_hand.items(): 
        q_final[reacher.data.actuator(name=key).id] = value
    
    traj_fun = create_quintic_traj_function(q_start, q_final, traj_time)
    traj_fun(traj_time)
    
    ctrl = MPPI(dynamics=dynamics_mpc_wrapper, 
            running_cost=runnng_cost, 
            nx = 68,
            num_samples = 1000,
            horizon=4,
            device="cpu",
            u_min=low_action,
            u_max=high_action,
            noise_sigma=noise_sigma,)
    
 
    if not os.path.exists("MPPI_" + obj_name + str(POSE_NUM) + ".npz"):
        # assuming you have a gym-like env
        
        action_seq = []
        costs_seq = []
        obs_mpc_state = reacher.reset_mpc()
        for i in range(MPC_STEPS):
            current_time = i*reacher.frame_skip*reacher.model.opt.timestep
            curent_q_CLEAR = traj_fun(current_time)[:,0]
            nominal_u = torch.zeros(ctrl.T, nu)
            for i_horizon in range(ctrl.T):
                current_time_horiz = (i+i_horizon)*reacher.frame_skip*reacher.model.opt.timestep
                curent_q = traj_fun(current_time_horiz)[:,0]
                nominal_u[i_horizon] = Tensor(curent_q)
            #ctrl.U = nominal_u
            #ctrl.u_init = initial_q
            #action_i = ctrl.command(obs_mpc_state, False)
            action_np = nominal_u[0]
            obs_mpc_state = reacher.step_mpc(obs_mpc_state, action_np)
            #reacher.step(action.cpu().numpy())
            cost = reacher.cost_mpc(obs_mpc_state, 0)
            costs_seq.append(cost)
            action_seq.append(action_np)
            print(f"Cost : {cost}")
            print(f"Action : {action_np}")
        reacher.close()
        print("Finish traj generate")
        np.savez("MPPI_"  + obj_name + str(POSE_NUM), action_seq)
    else:
        print("File traj already exists")
        action_seq = np.load("MPPI_" + obj_name + str(POSE_NUM) + ".npz")["arr_0"]
    obs_mpc_state = reacher.reset_mpc()
    #reacher.kinematics_debug = True

    
    reacher.kinematics_debug = False
    viewer = mujoco.viewer.launch_passive(reacher.model, reacher.data)
    while True:
        for action_i in action_seq:
            reacher.data.ctrl = action_i
            for i in range(reacher.frame_skip):
                mujoco.mj_step(reacher.model, reacher.data)
                for i in range(50):
                    time.sleep(0.001)
                    viewer.sync()
                    joint_id_x = reacher.data.model.joint(name="obj_t_joint_x").id
                    joint_id_y = reacher.data.model.joint(name="obj_t_joint_y").id
                    joint_id_z = reacher.data.model.joint(name="obj_t_joint_z").id
                    obj_speed_x = reacher.data.qvel[joint_id_x]
                    obj_speed_y = reacher.data.qvel[joint_id_y]
                    obj_speed_z = reacher.data.qvel[joint_id_z]
                    obj_speed = np.linalg.norm(np.array([obj_speed_x, obj_speed_y, obj_speed_z]))
                    distance_key_points_array, obj_speed, anglediff = reacher._get_obs()
                    print(f"Pose error: {np.sum(distance_key_points_array)}")


        for i in range(500):
            time.sleep(0.01)
            viewer.sync()
            mujoco.mj_step(reacher.model, reacher.data)
            if reacher.kinematics_debug:
                reacher.step(0)             
            joint_id_x = reacher.data.model.joint(name="obj_t_joint_x").id
            joint_id_y = reacher.data.model.joint(name="obj_t_joint_y").id
            joint_id_z = reacher.data.model.joint(name="obj_t_joint_z").id
            obj_speed_x = reacher.data.qvel[joint_id_x]
            obj_speed_y = reacher.data.qvel[joint_id_y]
            obj_speed_z = reacher.data.qvel[joint_id_z]
            obj_speed = np.linalg.norm(np.array([obj_speed_x, obj_speed_y, obj_speed_z]))
            distance_key_points_array, obj_speed, anglediff = reacher._get_obs()
            print(f"Pose error: {np.sum(distance_key_points_array)}")
           

          
            
        reacher.reset_mpc()
if __name__ == "__main__":
    run_mpc()