import glob
import os
import time
from icem_mpc.launch_colect import ConfigICEM, run_icem_from_config
from icem_mpc.mpc_utils import control_policy
from icem_mpc.reach_pose_env import ReachPoseEnv
import numpy as np

def random_point_on_sphere(radius=.4):
    # Случайный угол theta от 0 до 2π
    theta = 2 * np.pi * np.random.random()
    # Случайный угол phi от 0 до π
    phi = np.acos(2 * np.random.random() - 1)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.array([x, y, z])

file_name = "experts_traj_shadow_dexee/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03_POSENUM_0_dict_values__2_345___2_97___1_739__0_071__0_419__0_11__.npz"
load_file = np.load(file=file_name, allow_pickle=True, fix_imports=True)

pose_num = load_file["config_info"].item()["pose_num"]
name = load_file["config_info"].item()["obj_name"]
config = load_file["config_info"].item()["config"]
reward_dict = load_file["reward_dict"].item()

for _ in range(10):
    [x, y, z] = random_point_on_sphere()
    config.hand_starting_pose['WRJTx'] = x
    config.hand_starting_pose['WRJTy'] = y
    config.hand_starting_pose['WRJTz'] = z
    key_body_final_pos = load_file["config_info"].item()["key_body_final_pos"]
    env = ReachPoseEnv(config=config, reward_dict=reward_dict, render_mode="human", key_pose_dict=key_body_final_pos)

    control_policy(env, camera_name='mobile')