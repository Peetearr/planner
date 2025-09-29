import glob
import os
import time

import mujoco
from icem_mpc.launch_colect import ConfigICEM, run_icem_from_config
from icem_mpc.mpc_utils import trajectory_player
from icem_mpc.reach_pose_env import ReachPoseEnv
import numpy as np
import matplotlib.pyplot as plt
from icem_mpc.grasp_env_utils import set_position_kinematics




def show_final_position(reacher_pose_env: ReachPoseEnv):
    viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    dist_reward = []
    obj_displacement_reward = []
    wirst_orint_reward = []
    reacher_pose_env.kinematics_debug = True
    viewer.cam.distance = 1
    viewer.cam.elevation = -20
    viewer.cam.lookat[2] += -0.25

    for _ in range(20):
        for i in range(reacher_pose_env.frame_skip):
            
            mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
            for i in range(100):
                time.sleep(.02)
                viewer.sync()
            full_obs = reacher_pose_env._get_full_obs()
            rew, decompose = reacher_pose_env.reward(full_obs)
            dist_reward.append(decompose["distance_key_points"])
            obj_displacement_reward.append(decompose["obj_displacement"])
            wirst_orint_reward.append(decompose["diff_orient"])
            set_position_kinematics(reacher_pose_env.data, reacher_pose_env.hand_final_full_pose)
 
    viewer.close()



hand_name = "shadow_dexee"
folder = "experts_traj_" + hand_name
filenames = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.npz'))]
print(filenames)
for file_name in filenames:
    
    load_file = np.load(file=file_name, allow_pickle=True, fix_imports=True)
    pose_num = load_file["config_info"].item()["pose_num"]
    name = load_file["config_info"].item()["obj_name"]
    config = load_file["config_info"].item()["config"]
    reward_dict = load_file["reward_dict"].item()
    action_seq = load_file["action_seq"]
    key_body_final_pos = load_file["config_info"].item()["key_body_final_pos"]
    costs_seq = load_file["costs_seq"]
    ellites_trj = load_file["ellites_trj"]
    env = ReachPoseEnv(config=config, reward_dict=reward_dict, render_mode="human", key_pose_dict=key_body_final_pos)
    
    show_final_position(env)


