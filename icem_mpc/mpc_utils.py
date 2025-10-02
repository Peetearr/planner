import time
import mujoco
import numpy as np
from torch import Tensor
import torch
import matplotlib.pyplot as plt

from icem_mpc.reach_pose_env import ReachPoseEnv
from icem_mpc.bc import BC

import pickle

def camera_observation(model, data, camera_name="mobile"):
    renderer = mujoco.Renderer(model, 40, 60)
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera_name)
    pixels = renderer.render()
    pixels=pixels[:,:,0]
    mask_hand = 1.0*((pixels < 95) & (pixels > 0))
    mask_obj = 1.0*(pixels > 95)
    mask_hand[mask_hand==0] = 0
    mask_obj[mask_obj==0] = 0

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera_name)
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
    return demo_hand, demo_obj

def joints_observation():
    pass

def control_policy(reacher_pose_env: ReachPoseEnv, camera_name='joints', visualise=True, reward_flag=False):
    policy_path = 'model_weights_' + camera_name + '.pth'
    model = BC(4800*(camera_name!='joints') + 18*(camera_name=='joints'), 18)
    model.load_state_dict(torch.load(policy_path))
    model.eval()

    if visualise == True:
        viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    
        viewer.cam.distance = 1
        viewer.cam.elevation = -20
        viewer.cam.lookat[2] += -0.25

    renderer = mujoco.Renderer(reacher_pose_env.model, 40, 60)

    contact = 0
    while reacher_pose_env.data.time < .5:
        # get state
        if camera_name != 'joints':
            demo_hand, demo_obj = camera_observation(reacher_pose_env.model, reacher_pose_env.data, camera_name)
            obs = np.concatenate([demo_hand, demo_obj]).flatten()
        else:
            a = reacher_pose_env.data.qpos
            obs = np.concatenate([a[6:18], np.array([a[5], a[4], a[3], a[0], a[1], a[2]])])
        reacher_pose_env.data.ctrl = model.forward(torch.tensor(obs, dtype=torch.float32)).detach().numpy()

        for i in range(reacher_pose_env.data.ncon):
          geom1_id = reacher_pose_env.data.contact[i].geom1
          geom2_id = reacher_pose_env.data.contact[i].geom2

          body1_id = reacher_pose_env.model.geom_bodyid[geom1_id]
          body2_id = reacher_pose_env.model.geom_bodyid[geom2_id]

          body1_name = reacher_pose_env.model.body(body1_id).name
          body2_name = reacher_pose_env.model.body(body2_id).name
        #   print(body1_name, body2_name)

          if (body1_name == 'graspable_object' and body2_name != 'world') or \
          (body2_name == 'graspable_object' and body1_name != 'world'):
            contact += .002
        # dist = reacher_pose_env.reward(reacher_pose_env._get_full_obs(), reacher_pose_env.data.ctrl)[1]["distance_key_points"]
        # if max_reward < dist: max_reward = dist
        mujoco.mj_step(reacher_pose_env.model, reacher_pose_env.data)
        if visualise == True:
            viewer.sync()
    if visualise == True:
        viewer.close()
    if reward_flag:
        return contact



def trajectory_player(reacher_pose_env: ReachPoseEnv, action_seq, flying_camera = True, file_name = 'demos', cam=0):
    viewer = mujoco.viewer.launch_passive(reacher_pose_env.model, reacher_pose_env.data)
    dist_reward = []
    obj_displacement_reward = []
    wirst_orint_reward = []
    
    viewer.cam.distance = 1
    viewer.cam.elevation = -20
    viewer.cam.lookat[2] += -0.25

    demo = []
    cam_dict = ["mobile", "static"]
    with mujoco.Renderer(reacher_pose_env.model, 40, 60) as renderer:
        for action_i in action_seq:
            reacher_pose_env.data.ctrl = action_i
            ################
            if cam>=0:
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(reacher_pose_env.data, cam_dict[cam])
                pixels = renderer.render()
                pixels=pixels[:,:,0]
                mask_hand = 1.0*((pixels < 95) & (pixels > 0))
                mask_obj = 1.0*(pixels > 95)
                mask_hand[mask_hand==0] = 0
                mask_obj[mask_obj==0] = 0

                # depth & masking
                renderer.enable_depth_rendering()
                renderer.update_scene(reacher_pose_env.data, cam_dict[cam])
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


    viewer.close()
    if cam>=0:
        demos = {
            'observation': np.array(demo),
            'action': action_seq,
        }
        with open(file_name + '_' + cam_dict[cam] + '.pkl', 'wb') as f:
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
