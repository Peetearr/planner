import time
from config_crearor import prepare_env_config
from reach_pose_env import ReachPoseEnv


def run_env():
    POSE_NUM, obj_name, key_body_final_pos, config, final_act_pose_sh_hand = prepare_env_config()
    env = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human")

    state, info = env.reset()
    start_state = env.set_state(state[0], state[1])
    for _ in range(1000):
        time.sleep(0.01)
        reduced_obs, reward, _, _, debug_dict = env.step(env.action_space.sample())
        print(reward)


if __name__ == "__main__":
    run_env()
