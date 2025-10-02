import glob
import os
import time
from icem_mpc.launch_colect import ConfigICEM, run_icem_from_config
from icem_mpc.mpc_utils import control_policy
from icem_mpc.reach_pose_env import ReachPoseEnv
import numpy as np
import tqdm
import argparse

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import sample_farthest_points

def random_point_on_sphere(num=5):
    sphere = load_objs_as_meshes(["sphere.obj"])
    # dense_cloud
    dense_points = sample_points_from_meshes(
        sphere,
        num_samples=num*100,
        return_normals=False
    )

    # cloud
    points, _ = sample_farthest_points(
        dense_points,
        K=num,
        random_start_point=True
    )

    init_wirst_pose = {
        "WRJRx": 0,
        "WRJRy": 0,
        "WRJRz": 0,
        "WRJTx": 0,
        "WRJTy": 0,
        "WRJTz": 0,
    }
    pose_list = []
    for p in points[0]:
        x = p[0].float()
        y = p[1].float()
        z = p[2].float()
        init_wirst_pose["WRJTx"] = 1*x
        init_wirst_pose["WRJTy"] = 1*y
        init_wirst_pose["WRJTz"] = 1*z - .3

        n = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = -x/n, -y/n, -z/n
        init_wirst_pose["WRJRx"] = np.arctan2(-y,z)
        init_wirst_pose["WRJRy"] = np.arctan2(x,z)
        init_wirst_pose["WRJRz"] = np.arctan2(y,x)
        pose_list.append(init_wirst_pose.copy())
    return pose_list

parser = argparse.ArgumentParser()
parser.add_argument('--camera_name', default='mobile')
parser.add_argument('--num_init', default='15')
args = parser.parse_args()

camera_name = args.camera_name
file_name = "experts_traj_shadow_dexee/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03_POSENUM_0_dict_values__2_345___2_97___1_739__0_071__0_419__0_11__.npz"
load_file = np.load(file=file_name, allow_pickle=True, fix_imports=True)
num_init = int(args.num_init)

pose_num = load_file["config_info"].item()["pose_num"]
name = load_file["config_info"].item()["obj_name"]
config = load_file["config_info"].item()["config"]
reward_dict = load_file["reward_dict"].item()

init_pos_list = random_point_on_sphere(num_init)
result = []
for pose in tqdm.tqdm(init_pos_list):
    print(pose)
    config.hand_starting_pose = pose
    key_body_final_pos = load_file["config_info"].item()["key_body_final_pos"]
    env = ReachPoseEnv(config=config, reward_dict=reward_dict, render_mode="human", key_pose_dict=key_body_final_pos, disable_collide=False)

    final_ditance = control_policy(env, visualise=False, reward_flag=True, camera_name=camera_name)
    result.append(final_ditance)

np.save('result_' + camera_name + '.npy', result)
