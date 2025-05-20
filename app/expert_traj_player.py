import time
from icem_mpc.launch_colect import ConfigICEM, run_icem_from_config
from icem_mpc.mpc_utils import trajectory_player
from icem_mpc.reach_pose_env import ReachPoseEnv
import numpy as np
import reach_pose_env
import matplotlib.pyplot as plt

file_name = "experts_traj/core-bowl-a593e8863200fdb0664b3b9b23ddfcbc/core-bowl-a593e8863200fdb0664b3b9b23ddfcbc_POSENUM_0_dict_values__1_571__0_0__0_0__0_0__0_4__0_0__.npz"

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
config_icem.mpc_steps = 30
config_icem.warmup_iters = 50
config_icem.online_iters = 50
config_icem.num_samples = 30

config_icem.num_elites = 10
config_icem.elites_keep_fraction = 0.5
config_icem.alpha = 0.003

config_icem.num_samples_after_reset = 60
config_icem.reset_penalty_thr = -0.5
config_icem.num_elites_after_reset = 20


start_time = time.time()
new_action_seq, new_costs_seq, new_full_observations, new_ellites_trj = run_icem_from_config(
    reward_dict, config_icem, key_body_final_pos, config, render_mode="human"
)
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")



 


trajectory_player(env, new_action_seq)
