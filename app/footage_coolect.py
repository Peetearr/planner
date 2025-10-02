import glob
import os
import time
from icem_mpc.launch_colect import ConfigICEM, run_icem_from_config
from icem_mpc.mpc_utils import trajectory_player
from icem_mpc.reach_pose_env import ReachPoseEnv
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_camera', default='0')
args = parser.parse_args()

n_cam = int(args.n_camera)
hand_name = "shadow_dexee"
folder = "experts_traj_" + hand_name + "/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/valid_traj"
# folder = "single_runner/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03"

filenames = glob.glob(os.path.join(folder, '*.npz'))

for file_name in filenames:
    if not os.path.exists(file_name.replace('.npz', '.pkl')):
        print(file_name)
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

        trajectory_player(env, action_seq,flying_camera=False, file_name=file_name[:-4], cam=1)