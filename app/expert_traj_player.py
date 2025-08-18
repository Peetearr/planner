import glob
import os
import time
from icem_mpc.launch_colect import ConfigICEM, run_icem_from_config
from icem_mpc.mpc_utils import trajectory_player
from icem_mpc.reach_pose_env import ReachPoseEnv
import numpy as np

import matplotlib.pyplot as plt

file_name = "experts_traj_shadow_dexee/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03_POSENUM_6_dict_values__1_571__0_0__0_0___0_3__0_1__0_0__.npz"

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

config_icem = ConfigICEM()
config_icem.horizon = 10
config_icem.mpc_steps = 27
config_icem.warmup_iters = 120
config_icem.online_iters = 50
config_icem.num_samples = 30

config_icem.num_elites = 10
config_icem.elites_keep_fraction = 0.5
config_icem.alpha = 0.003

config_icem.num_samples_after_reset = 100
config_icem.reset_penalty_thr = -0.8
config_icem.num_elites_after_reset = 60

trajectory_player(env, action_seq)
