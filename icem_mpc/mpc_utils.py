import time
import mujoco
import numpy as np
from torch import Tensor
import torch
import matplotlib.pyplot as plt

from icem_mpc.reach_pose_env import ReachPoseEnv
from icem_mpc.bc import BC

import pickle

def control_policy(reacher_pose_env: ReachPoseEnv, policy_path='model_weights.pth'):
    model = BC(4800, 18)
    model.load_state_dict(torch.load(policy_path))
    model.eval()

    viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    
    viewer.cam.distance = 1
    viewer.cam.elevation = -20
    viewer.cam.lookat[2] += -0.25

    renderer = mujoco.Renderer(reacher_pose_env.model, 40, 60)
    while reacher_pose_env.data.time < .5:
        # get state
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(reacher_pose_env.data, "mobile")
        pixels = renderer.render()
        pixels=pixels[:,:,0]
        mask_hand = 1.0*((pixels < 95) & (pixels > 0))
        mask_obj = 1.0*(pixels > 95)
        mask_hand[mask_hand==0] = 0
        mask_obj[mask_obj==0] = 0

        renderer.enable_depth_rendering()
        renderer.update_scene(reacher_pose_env.data, "mobile")
        pixels = renderer.render()
        demo_hand = pixels * mask_hand
        demo_obj = pixels * mask_obj
        try:
            demo_obj[demo_obj == 0] = np.unique(demo_obj)[1]
            demo_obj -= demo_obj.min()
            demo_obj = demo_obj/(demo_obj.max() + 1e-10)
        except:
            pass

        try:
            demo_hand[demo_hand == 0] = np.unique(demo_hand)[1]
            demo_hand -= demo_hand.min()
            demo_hand = demo_hand/(demo_hand.max() + 1e-10)
        except:
            pass

        obs = np.concatenate([demo_hand, demo_obj]).flatten()
        reacher_pose_env.data.ctrl = model.forward(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
        mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
        time.sleep(0.02)
        viewer.sync()
    viewer.close()



def trajectory_player(reacher_pose_env: ReachPoseEnv, action_seq, flying_camera = True, file_name = 'demos'):
    viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    dist_reward = []
    obj_displacement_reward = []
    wirst_orint_reward = []
    
    viewer.cam.distance = 1
    viewer.cam.elevation = -20
    viewer.cam.lookat[2] += -0.25

    demo = []
    with mujoco.Renderer(reacher_pose_env.model, 40, 60) as renderer:
        for action_i in action_seq:
            reacher_pose_env.data.ctrl = action_i
            ################
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(reacher_pose_env.data, "mobile")
            pixels = renderer.render()
            pixels=pixels[:,:,0]
            mask_hand = 1.0*((pixels < 95) & (pixels > 0))
            mask_obj = 1.0*(pixels > 95)
            mask_hand[mask_hand==0] = 0
            mask_obj[mask_obj==0] = 0

            # depth & masking
            renderer.enable_depth_rendering()
            renderer.update_scene(reacher_pose_env.data, "mobile")
            pixels = renderer.render()
            demo_hand = pixels * mask_hand
            demo_obj = pixels * mask_obj
            demo_hand[demo_hand == 0] = np.unique(demo_hand)[1]
            try:
                demo_obj[demo_obj == 0] = np.unique(demo_obj)[1]
                demo_obj -= demo_obj.min()
                demo_obj = demo_obj/(demo_obj.max() + 1e-10)
            except:
                pass

            demo_hand -= demo_hand.min()
            demo_hand = demo_hand/(demo_hand.max() + 1e-10)
            demo.append(np.concatenate([demo_hand, demo_obj]).flatten())
            ################
            for i in range(reacher_pose_env.frame_skip):
                mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
                for i in range(25):
                    time.sleep(0.002)
                    if flying_camera:
                        viewer.cam.elevation += 0.005
                    viewer.sync()
                full_obs = reacher_pose_env._get_full_obs()
                rew, decompose = reacher_pose_env.reward(full_obs, action_i)
                dist_reward.append(decompose["distance_key_points"])
                obj_displacement_reward.append(decompose["obj_displacement"])
                wirst_orint_reward.append(decompose["diff_orient"])

            viewer.cam.trackbodyid = reacher_pose_env.data.model.body(name="graspable_object").id


    # reacher_pose_env.kinematics_debug = False
    # for i in range(1000):
    #     time.sleep(0.01)
    #     viewer.sync()
    #     if reacher_pose_env.kinematics_debug:
    #         reacher_pose_env.render_mode = None
    #         reacher_pose_env.step(0)
    #     else:
    #         if flying_camera:
    #             viewer.cam.azimuth += 0.4
    viewer.close()
    demos = {
        'observation': np.array(demo),
        'action': action_seq,
    }
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(demos, f)


    # plt.figure()

    # plt.plot(dist_reward, label="r1")
    # plt.xlabel("Time step")
    # plt.ylabel("Distance reward")

    # plt.figure()

    # plt.plot(obj_displacement_reward, label="r2")
    # plt.xlabel("Time step")
    # plt.ylabel("Obj displacement reward")

    # plt.figure()

    # plt.plot(wirst_orint_reward, label="r3")
    # plt.xlabel("Time step")
    # plt.ylabel("wirst_orint_reward")
    # plt.show()


def dynamics_mpc_wrapper_np(env: ReachPoseEnv, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    res_states = np.zeros(state_vec.shape, dtype=state_vec.dtype)
    for state, act, i in zip(state_vec, action_vec, range(state_vec.shape[0])):
        env.set_state(state)
        reduced_obs, reward, _, _, debug_dict = env.step(act)
        state = debug_dict["state"]
        res_states[i, :] = state
    return res_states


def dynamics_mpc_wrapper_tensor(env: ReachPoseEnv, state_vec: Tensor, action_vec: Tensor):
    res_states = dynamics_mpc_wrapper_np(env, state_vec.cpu().numpy(), action_vec.cpu().numpy())
    res_states_tensor = Tensor(res_states)
    return res_states_tensor


def cost_traj_mpc_np(env: ReachPoseEnv, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
    costs = np.zeros(state_vec.shape[0], dtype=np.float32)
    for trj, act_horizon, i in zip(state_vec, action_vec, range(state_vec.shape[0])):
        accum_cost_trj = 0
        for state_i, act_i in zip(trj, act_horizon):
            env.set_state(state_i)
            obs = env._get_full_obs()
            step_cost, _ = env.reward(obs, act_i)
            accum_cost_trj += -step_cost
        costs[i] = accum_cost_trj

    return costs


def cost_traj_mpc_tensor(env: ReachPoseEnv, state_vec: Tensor, action_vec: Tensor) -> Tensor:
    costs = cost_traj_mpc_np(env, state_vec.cpu().numpy(), action_vec.cpu().numpy())
    costs_tensor = Tensor(costs)
    return costs_tensor
