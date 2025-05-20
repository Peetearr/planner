from dataclasses import dataclass
import itertools
import os
import re
import time
from typing import Optional
import numpy as np

from pytorch_icem import iCEM
from torch import Tensor
from functools import partial
from tqdm import tqdm

from icem_mpc.config_crearor import get_tabale_top_start_pos, prepare_env_config
from icem_mpc.mpc_utils import trajectory_player, cost_traj_mpc_tensor, dynamics_mpc_wrapper_tensor
from icem_mpc.reach_pose_env import ReachPoseEnv, ReachPoseEnvConfig


@dataclass
class ConfigICEM:
    warmup_iters: int = 25
    online_iters: int = 25
    num_samples: int = 250
    num_elites: int = 20
    elites_keep_fraction: float = 0.5
    horizon: int = 7
    alpha: float = 0.005
    mpc_steps: int = 40
    num_samples_after_reset: int = 250
    reset_penalty_thr: float = -0.8
    num_elites_after_reset: int = 60


def run_icem(
    obj_name: str,
    pose_num: int,
    reward_dict: dict[str, float],
    confi_icem: ConfigICEM,
    frame_skip: int = 4,
    render_mode: Optional[str] = None,
):
    POSE_NUM, obj_name, key_body_final_pos, config, final_hand_joint_pose = prepare_env_config(
        obj_name=obj_name, pose_num=pose_num, frame_skip=frame_skip
    )
    action_seq, costs_seq, full_observations, ellites_trj = run_icem_from_config(
        reward_dict,
        confi_icem,
        key_body_final_pos,
        config,
        render_mode,
    )

    return action_seq, costs_seq, ellites_trj, full_observations


def run_icem_from_config(
    reward_dict: dict[str, float],
    confi_icem: ConfigICEM,
    key_body_final_pos: dict[str, np.ndarray],
    config: ReachPoseEnvConfig,
    render_mode: Optional[str] = None,
):
    reacher = ReachPoseEnv(
        config,
        key_pose_dict=key_body_final_pos,
        render_mode=render_mode,
        reward_dict=reward_dict,
    )
    reacher_dynamic = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, reward_dict=reward_dict)

    vector_dynamics = partial(dynamics_mpc_wrapper_tensor, reacher_dynamic)
    vector_cost = partial(cost_traj_mpc_tensor, reacher_dynamic)

    initial_mean = (reacher.action_space.low + reacher.action_space.high) / 2
    initial_sigma = reacher.action_space.high - initial_mean

    start_state = reacher.get_state()
    nu = reacher.action_space.shape[0]

    ctrl = iCEM(
        dynamics=vector_dynamics,
        trajectory_cost=vector_cost,
        sigma=Tensor(initial_sigma),
        nx=start_state.shape[0],
        nu=nu,
        warmup_iters=confi_icem.warmup_iters,
        online_iters=confi_icem.online_iters,
        num_samples=confi_icem.num_samples,
        num_elites=confi_icem.num_elites,
        elites_keep_fraction=confi_icem.elites_keep_fraction,
        horizon=confi_icem.horizon,
        device="cpu",
        alpha=confi_icem.alpha,
        noise_beta=2,
        low_bound_action=reacher.action_space.low,
        high_bound_action=reacher.action_space.high,
    )

    ctrl.mean = Tensor(initial_mean)
    ctrl.sigma = Tensor(initial_sigma)

    action_seq = np.zeros((confi_icem.mpc_steps, nu), dtype=np.float32)
    costs_seq = np.zeros(confi_icem.mpc_steps)
    full_observations = []
    obs_mpc_state = start_state

    number_of_trj = int(ctrl.keep_fraction * ctrl.K)
    ellites_trj = np.zeros((number_of_trj, confi_icem.mpc_steps, nu), dtype=np.float32)
    is_reseted = False

    for i in range(confi_icem.mpc_steps):
        action_i = ctrl.command(obs_mpc_state, shift_nominal_trajectory=False)
        action_np = action_i.cpu().numpy()
        reduced_obs, reward, _, _, debug_dict = reacher.step(action_np)
        obs_mpc_state = reacher.get_state()

        # Collect data
        costs_seq[i] = reward
        action_seq[i] = action_np
        ellites_trj[:, i, :] = ctrl.kept_elites[:number_of_trj, 0, :].to("cpu").numpy()
        full_observations.append(debug_dict["full_obs"])

        # Debugging
        if reacher.render_mode == "human":
            print(f"Cost : {reward}")
            print(f"Step : {i}")
            mean_normilize = ctrl.mean[0]
            std_normilize = ctrl.std[0] / initial_sigma
            print(f"MEAN[0] {np.array(mean_normilize).round(3)}")
            print()
            print(f"STD[0] {np.array(std_normilize).round(3)}")

        ctrl.shift()

        if reward < confi_icem.reset_penalty_thr and not is_reseted:
            ctrl.reset()
            ctrl.N = confi_icem.num_samples_after_reset
            ctrl.K = confi_icem.num_elites_after_reset
            is_reseted = True
            ctrl.alpha = 0.0015

    if reacher.render_mode == "human":
        reacher.close()
    return action_seq, costs_seq, full_observations, ellites_trj


def create_configs_for_env(obj_path: str, pose_nums: list[int] = [0, 1, 2]):
    start_poses, x_pose, y_pose = get_tabale_top_start_pos()
    start_p_num = itertools.product(start_poses, pose_nums)
    env_config_list = []
    for start_pose_hand_i, pose_num_i in start_p_num:
        pose_num, obj_name, key_body_final_pos, config, final_act_pose_sh_hand = prepare_env_config(
            pose_num=pose_num_i, obj_name=obj_path
        )
        config.hand_starting_pose = start_pose_hand_i
        conif_dict = {
            "hand_starting_pose": start_pose_hand_i,
            "obj_name": obj_name,
            "pose_num": pose_num_i,
            "key_body_final_pos": key_body_final_pos,
            "final_act_pose_sh_hand": final_act_pose_sh_hand,
            "config": config,
        }
        env_config_list.append((conif_dict))
    return env_config_list


def create_file_name_based_position(
    obj_name: str, pose_num: int, hand_starting_pose: dict[str, float], folder="experts_traj"
):
    hand_starting_pose_name = {k: round(float(v), 3) for k, v in hand_starting_pose.items()}
    start_pos_name = re.sub(r"[^a-zA-Z0-9]", "_", str(hand_starting_pose_name.values()))
    subfolder = os.path.join(folder, obj_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    filename = f"{obj_name}_POSENUM_{pose_num}_{start_pos_name}.npz"
    filename = os.path.join(subfolder, filename)
    if os.path.exists(filename):
        print("File already exists")
    return filename


def run_object_run(
    reward_dict: dict,
    config_icem: ConfigICEM,
    pose_nums: list[int] = [0, 1, 2],
    obj_name: str = "core-bowl-a593e8863200fdb0664b3b9b23ddfcbc",
):

    configs_and_info = create_configs_for_env(obj_name, pose_nums)

    start_time = time.time()
    with tqdm(total=len(configs_and_info), desc="Processing iCEM MPC") as pbar:
        for config_i in configs_and_info:
            action_seq, costs_seq, full_observations, ellites_trj = run_icem_from_config(
                reward_dict=reward_dict,
                confi_icem=config_icem,
                key_body_final_pos=config_i["key_body_final_pos"],
                config=config_i["config"],
            )
            pbar.update(1)
            file_name = create_file_name_based_position(
                obj_name=obj_name, pose_num=config_i["pose_num"], hand_starting_pose=config_i["hand_starting_pose"]
            )
            np.savez(
                file_name,
                action_seq=action_seq,
                costs_seq=costs_seq,
                full_observations=full_observations,
                ellites_trj=ellites_trj,
                config_info=config_i,
                reward_dict=reward_dict,
            )
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":

    reward_dict = {
        "distance_key_points": 1.5,
        "obj_displacement": 2.0,
        "diff_orient": 1.0,
        "obj_speed": 1.0,
    }

    config_icem = ConfigICEM()
    config_icem.horizon = 10
    config_icem.mpc_steps = 23
    config_icem.warmup_iters = 120
    config_icem.online_iters = 50
    config_icem.num_samples = 30

    config_icem.num_elites = 20
    config_icem.elites_keep_fraction = 0.1
    config_icem.alpha = 0.003

    config_icem.num_samples_after_reset = 250
    config_icem.reset_penalty_thr = -0.8
    config_icem.num_elites_after_reset = 60

    run_object_run(reward_dict, config_icem, obj_name="core-bowl-a593e8863200fdb0664b3b9b23ddfcbc", pose_nums=[0, 1])
