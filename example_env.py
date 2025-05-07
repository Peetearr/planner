import time
from config_crearor import get_tabale_top_start_pos, prepare_env_config
from reach_pose_env import ReachPoseEnv
import numpy as np


def run_env():
    POSE_NUM, obj_name, key_body_final_pos, config, final_act_pose_sh_hand = prepare_env_config()

    start_poses, x_pose, y_pose = get_tabale_top_start_pos()

    env = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human")

    for _ in range(1000):
        time.sleep(0.01)

        reduced_obs, reward, _, _, debug_dict = env.step(np.zeros(len(env.action_space.sample())))
        print(reward)


if __name__ == "__main__":
    run_env()
