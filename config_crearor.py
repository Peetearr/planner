from copy import deepcopy
import numpy as np
import transforms3d.euler as euler
from grasp_env_utils import (
    default_mapping,
    get_final_bodies_pose,
    transform_wirst_pos_to_obj,
)

from reach_pose_env import ReachPoseEnvConfig, convert_pose_dexgraspnet_to_mujoco


def prepare_env_config(frame_skip=4, pose_num=10, obj_name="sem-Plate-9969f6178dcd67101c75d484f9069623"):

    pos_path_name = "final_positions/" + obj_name + ".npy"
    mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"
    model_path_hand = "./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"

    obj_quat_good = euler.euler2quat(np.deg2rad(90), np.deg2rad(180), np.deg2rad(180))
    obj_pos_good = [0.0, 0, 0.4]

    obj_start_pos = np.array([0.0, 0, -0.3])

    obj_start_quat = euler.euler2quat(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))

    core_mug = np.load(pos_path_name, allow_pickle=True)
    qpos_hand = core_mug[pose_num]["qpos"]

    wirst_pos = np.array(
        [
            qpos_hand["WRJTx"],
            qpos_hand["WRJTy"],
            qpos_hand["WRJTz"],
        ]
    )
    wirst_quat = euler.euler2quat(
        qpos_hand["WRJRx"],
        qpos_hand["WRJRy"],
        qpos_hand["WRJRz"],
    )

    transformed_wirst_pose, transformed_rot = transform_wirst_pos_to_obj(
        obj_start_pos, obj_start_quat, wirst_pos, wirst_quat
    )

    transformed_wirst_euler_ang = euler.mat2euler(transformed_rot)

    new_wirst_pos = {
        "WRJRx": transformed_wirst_euler_ang[0],
        "WRJRy": transformed_wirst_euler_ang[1],
        "WRJRz": transformed_wirst_euler_ang[2],
        "WRJTx": transformed_wirst_pose[0],
        "WRJTy": transformed_wirst_pose[1],
        "WRJTz": transformed_wirst_pose[2],
    }

    for key in new_wirst_pos.keys():
        qpos_hand[key] = new_wirst_pos[key]

    final_act_pose_sh_hand = convert_pose_dexgraspnet_to_mujoco(qpos_hand, default_mapping)
    key_body_final_pos = get_final_bodies_pose(qpos_hand, model_path_hand)

    init_wirst_pose = {
        "WRJRx": np.deg2rad(90),
        "WRJRy": 0,
        "WRJRz": 0,
        "WRJTx": 0,
        "WRJTy": 0,
        "WRJTz": 0.2,
    }

    config = ReachPoseEnvConfig(
        hand_final_full_pose=qpos_hand,
        model_path_hand=model_path_hand,
        obj_mesh_path=mesh_path,
        obj_start_pos=obj_start_pos,
        obj_start_quat=obj_start_quat,
        obj_scale=core_mug[pose_num]["scale"] * 0.9,
        frame_skip=frame_skip,
        hand_starting_pose=init_wirst_pose,
    )

    return pose_num, obj_name, key_body_final_pos, config, final_act_pose_sh_hand


def get_tabale_top_start_pos():
    offset_x = 0
    offset_y = 0.4
    square_size = 0.3
    positions = 3
    posible_x = np.linspace(-square_size + offset_x  , square_size + offset_x, positions)
    posible_y = np.linspace(-square_size + offset_y, square_size + offset_y, positions)
    xy_postions = np.meshgrid(posible_x, posible_y)

    init_wirst_pose = {
        "WRJRx": np.deg2rad(90),
        "WRJRy": 0,
        "WRJRz": 0,
        "WRJTx": 0,
        "WRJTy": 0,
        "WRJTz": 0,
    }
    pose_list = []
    for x, y in zip(xy_postions[0].flatten(), xy_postions[1].flatten()):
        init_wirst_pose["WRJTx"] = x
        init_wirst_pose["WRJTy"] = y
        pos = deepcopy(init_wirst_pose)
        pose_list.append(pos)
    return pose_list, xy_postions[0].flatten(), xy_postions[1].flatten()
 
