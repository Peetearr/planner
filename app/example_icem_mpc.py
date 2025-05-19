import os
import time
import numpy as np
from pytorch_icem import iCEM
from torch import Tensor
from functools import partial
import re
from icem_mpc.mpc_utils import dynamics_mpc_wrapper_tensor, cost_traj_mpc_tensor, trajectory_player
from icem_mpc.reach_pose_env import ReachPoseEnv, revert_obs_vector
from icem_mpc.config_crearor import prepare_env_config


def run_mpc():
    POSE_NUM, obj_name, key_body_final_pos, config, final_hand_joint_pose = prepare_env_config(
        obj_name="core-bowl-a593e8863200fdb0664b3b9b23ddfcbc", pose_num=4
    )

    reward_dict = {
        "distance_key_points": 1.5,
        "obj_displacement": 2.0,
        "diff_orient": 1.0,
        "obj_speed": 1.0,
    }
    reacher = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human", reward_dict=reward_dict)
    reacher_dynamic = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, reward_dict=reward_dict)

    vector_dynamics = partial(dynamics_mpc_wrapper_tensor, reacher_dynamic)
    vector_cost = partial(cost_traj_mpc_tensor, reacher_dynamic)

    _, info = reacher.reset()
    start_state = reacher.get_state()
    nu = reacher.action_space.shape[0]

    initial_mean = (reacher.action_space.low + reacher.action_space.high) / 2
    initial_sigma = reacher.action_space.high - initial_mean

    ctrl = iCEM(
        dynamics=vector_dynamics,
        trajectory_cost=vector_cost,
        sigma=Tensor(initial_sigma),
        nx=start_state.shape[0],
        nu=nu,
        warmup_iters=120,
        online_iters=50,
        num_samples=30,
        num_elites=20,
        elites_keep_fraction=0.1,
        horizon=10,
        device="cpu",
        alpha=0.003,
        noise_beta=2,
        low_bound_action=reacher.action_space.low,
        high_bound_action=reacher.action_space.high,
    )

    ctrl.mean = Tensor(initial_mean)
    ctrl.sigma = Tensor(initial_sigma)
    is_reseted = False
    # Convert all values inside reacher.hand_starting_pose to Python float with round 3
    hand_starting_pose_name = {k: round(float(v), 3) for k, v in reacher.hand_starting_pose.items()}
    filename = (
        f"{obj_name}_POSENUM_{POSE_NUM}_{re.sub(r'[^a-zA-Z0-9]', '_', str(hand_starting_pose_name.values()))}.npz"
    )
    if not os.path.exists(filename):
        start_time = time.time()
        MPC_STEPS = 25
        action_seq = np.zeros((MPC_STEPS, nu), dtype=np.float32)
        costs_seq = np.zeros(MPC_STEPS)
        full_observations = []
        obs_mpc_state = start_state

        number_of_trj = int(ctrl.keep_fraction * ctrl.K)
        ellites_trj = np.zeros((number_of_trj, MPC_STEPS, nu), dtype=np.float32)

        for i in range(MPC_STEPS):
            action_i = ctrl.command(obs_mpc_state, shift_nominal_trajectory=False)
            action_np = action_i.cpu().numpy()
            reduced_obs, reward, _, _, debug_dict = reacher.step(action_np)
            obs_mpc_state = reacher.get_state()

            costs_seq[i] = reward
            action_seq[i] = action_np
            ellites_trj[:, i, :] = ctrl.kept_elites[:number_of_trj, 0, :].to("cpu").numpy()
            full_observations.append(debug_dict["full_obs"])
            print(f"Cost : {reward}")
            print(f"Step : {i}")
            mean_normilize = ctrl.mean[0]
            std_normilize = ctrl.std[0] / initial_sigma
            print(f"MEAN[0] {np.array(mean_normilize).round(3)}")
            print()
            print(f"STD[0] {np.array(std_normilize).round(3)}")

            ctrl.shift()
            if reward > -0.8 and not is_reseted:
                ctrl.reset()
                print("Reset")
                ctrl.N = 250
                ctrl.K = 60
                is_reseted = True
                ctrl.alpha = 0.0015
        reacher.close()
        print("Finish traj generate")
        print("Time elapsed: ", time.time() - start_time)

        np.savez(
            filename,
            action_seq=action_seq,
            elites_action=ellites_trj,
            costs_seq=costs_seq,
            full_observations=full_observations,
        )

        # np.savez(obj_name + str("_cost_") + str(POSE_NUM), costs_seq)
    else:
        print("File traj already exists")
        action_seq = np.load(filename)["action_seq"]
    obs_mpc_state = reacher.reset()

    trajectory_player(reacher, action_seq)


if __name__ == "__main__":
    run_mpc()
