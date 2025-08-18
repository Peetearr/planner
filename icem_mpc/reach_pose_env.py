from copy import deepcopy
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple, Union
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import transforms3d.euler as euler
from transforms3d import quaternions
from numpy.typing import NDArray
from dataclasses import dataclass, field

from icem_mpc.grasp_env_utils import (
    get_key_bodies_pose,
    set_position_kinematics,
)

from icem_mpc.load_complex_obj import add_graspable_body, add_meshes_from_folder


def revert_obs_vector(
    flat_obs: NDArray,
    keys: List[str],
    value_info: List[Tuple[int, str, Union[int, Tuple[int, ...]]]],
) -> Dict[str, Any]:
    """
    Revert the effect of get_obs_vector by reconstructing the original obs_dict from a flattened NDArray.

    Args:
        flat_obs (NDArray): The flattened observation vector produced by get_obs_vector.
        keys (List[str]): Sorted list of keys from the original obs_dict.
        value_info (List[Tuple[int, str, Union[int, Tuple[int, ...]]]]): Metadata for each key, where each tuple contains:
            - size (int): Number of elements in the flattened value.
            - type (str): Either "float" (for single float) or "array" (for np.ndarray).
            - shape (Union[int, Tuple[int, ...]]): Original shape of the value (int for float, tuple for array).

    Returns:
        Dict[str, Any]: Reconstructed obs_dict with keys mapping to floats or np.ndarrays.

    Raises:
        ValueError: If the flat_obs length does not match the sum of value sizes or if metadata is invalid.
    """
    # Validate inputs
    if len(keys) != len(value_info):
        raise ValueError("Number of keys must match number of value_info entries")

    # Calculate total expected size
    total_size = sum(info[0] for info in value_info)
    if flat_obs.size != total_size:
        raise ValueError(f"Flat observation size ({flat_obs.size}) does not match expected size ({total_size})")

    # Reconstruct the dictionary
    obs_dict: Dict[str, Any] = {}
    current_idx = 0

    for key, (size, val_type, shape) in zip(keys, value_info):
        # Extract the segment for this key
        segment = flat_obs[current_idx : current_idx + size]

        if val_type == "float":
            # For floats, expect a single value
            if size != 1:
                raise ValueError(f"Float value for key '{key}' must have size 1, got {size}")
            if not isinstance(shape, int) or shape != 1:
                raise ValueError(f"Shape for float key '{key}' must be 1, got {shape}")
            obs_dict[key] = float(segment[0])

        elif val_type == "array":
            # For arrays, reshape to the original shape
            if not isinstance(shape, tuple):
                raise ValueError(f"Shape for array key '{key}' must be a tuple, got {shape}")
            try:
                obs_dict[key] = segment.reshape(shape)
            except ValueError as e:
                raise ValueError(f"Cannot reshape segment for key '{key}' to shape {shape}: {e}")

        else:
            raise ValueError(f"Invalid type '{val_type}' for key '{key}', must be 'float' or 'array'")

        current_idx += size

    return obs_dict


@dataclass
class ReachPoseEnvConfig:
    model_path_hand: str
    obj_mesh_path: str
    hand_final_full_pose: Dict[str, float]
    obj_start_pos: np.ndarray
    obj_start_quat: np.ndarray
    obj_scale: float = 1.0
    frame_skip: int = 5
    hand_starting_pose: Dict[str, float] = field(default_factory=dict)

    def __hash__(self):
        # Convert dictionaries to tuples of sorted key-value pairs for hashing
        hand_final_pose_tuple = tuple(sorted(self.hand_final_full_pose.items()))
        hand_starting_pose_tuple = tuple(sorted(self.hand_starting_pose.items()))
        # Convert NumPy arrays to tuples for hashing
        obj_start_pos_tuple = tuple(self.obj_start_pos.flatten())
        obj_start_quat_tuple = tuple(self.obj_start_quat.flatten())
        # Combine all fields into a tuple and hash
        return hash(
            (
                self.model_path_hand,
                self.obj_mesh_path,
                hand_final_pose_tuple,
                obj_start_pos_tuple,
                obj_start_quat_tuple,
                self.obj_scale,
                self.frame_skip,
                hand_starting_pose_tuple,
            )
        )


def set_position(mj_data: mujoco.MjData, qpos: dict[str, float], maping: dict[str, str] = None):
    if maping is None:
        for key, value in qpos.items():
            mj_data.actuator(key).ctrl = value
    else:
        for key, value in qpos.items():
            mj_data.actuator(maping[key]).ctrl = value


def convert_pose_dexgraspnet_to_mujoco(qpos: dict[str, float], maping: dict[str, str] = None):
    new_qpos = {}
    if maping is None:
        for key, value in qpos.items():
            qpos[key] = qpos[key]
    else:
        for key, value in qpos.items():
            new_qpos[maping[key]] = qpos[key]

    return new_qpos


class ReachPoseEnv(MujocoEnv):
    def __init__(
        self,
        config: ReachPoseEnvConfig,
        reward_dict: Optional[dict] = None,
        render_mode: Optional[str] = None,
        width=640,
        height=480,
        key_pose_dict: Optional[dict] = None,
        **kwargs,
    ):
        self.obj_scale = config.obj_scale
        self.obj_start_pos = config.obj_start_pos
        self.obj_start_quat = config.obj_start_quat
        self.obj_mesh_path = config.obj_mesh_path
        self.hand_final_full_pose = config.hand_final_full_pose
        if config.hand_starting_pose is not None:
            self.hand_starting_pose = config.hand_starting_pose
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.key_pose_dict = key_pose_dict
        self.kinematics_debug = False
        if reward_dict is None:
            self.reward_dict = {
                "distance_key_points": 1,
                "obj_displacement": 1,
                "diff_orient": 1,
            }
        else:
            self.reward_dict = reward_dict
        self.tip_body_names = [
            "F0/finger_distal",
            "F1/finger_distal",
            "F2/finger_distal"
        ]
        self.palm_name = "hand_base"
        self.action_space = self.init_action_space(config.model_path_hand)

        super().__init__(
            model_path=config.model_path_hand,
            frame_skip=config.frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=width,
            height=height,
            **kwargs,
        )

    def add_body_key_points(self):
        for pose_name, pose in self.key_pose_dict.items():
            self.spec_mujoco.worldbody.add_geom(
                name=pose_name + "_ball",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=[1, 1, 0.1, 0.5],
                size=[0.003, 0.003, 0.003],
                pos=pose,
                contype=0,
                conaffinity=0,
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        state, info = super().reset(seed=seed)
        state = self.get_state()
        return state, info

    def reset_model(self) -> np.ndarray:
        """
        Reset the state of the environment to an initial state.
        """

        self.set_state(self.init_qpos, self.init_qvel)

        return np.array(self.init_qpos, dtype=np.float32)

    def init_action_space(self, mj_path):
        """
        Determine the action space based on the mujoco model
        """
        self.model = mujoco.MjModel.from_xml_path(mj_path)
        bounds = super()._set_action_space()
        return bounds

    def _initialize_simulation(self):

        spec_mujoco = mujoco.MjSpec.from_file(self.fullpath)
        self.spec_mujoco = spec_mujoco

        self.add_body_key_points()

        self.simultion_settings(spec_mujoco)
        self.add_graspable_object_spec(spec_mujoco)

        composite_model = spec_mujoco.compile()
        composite_data = mujoco.MjData(composite_model)

        if self.hand_starting_pose is not None:
            set_position_kinematics(composite_data, self.hand_starting_pose)
        # self.s
        composite_model.vis.global_.offwidth = self.width
        composite_model.vis.global_.offheight = self.height

        self.init_qpos = composite_data.qpos.ravel().copy()
        self.init_qvel = composite_data.qvel.ravel().copy()

        return composite_model, composite_data

    def simultion_settings(self, spec: mujoco.MjSpec):
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT  # type: ignore
        # spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        spec.option.timestep = 0.0025

    def add_graspable_object_spec(self, spec):
        combined_mesh, mesh_names = add_meshes_from_folder(
            spec,
            self.obj_mesh_path,
            prefix="obj_",
            scale=[self.obj_scale, self.obj_scale, self.obj_scale],
        )
        self.add_grapable_obj(spec, combined_mesh, mesh_names)

        spec.find_body("graspable_object").pos = self.obj_start_pos
        spec.find_body("graspable_object").quat = self.obj_start_quat

    # fmt: off
    def add_grapable_obj(self, spec, combined_mesh, mesh_names):
        graspable_body = add_graspable_body(spec, combined_mesh, mesh_names, density=(1000 * 0.14))

        for gg in graspable_body.geoms:
            gg.friction = [0.8, 0.009, 0.0001]
        
        friction_loss = 3
        damping = 2

        graspable_body.gravcomp = 1
        graspable_body.add_joint(
            name="obj_t_joint_x", axis=[1, 0, 0], frictionloss=friction_loss, damping=damping,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
        )
        graspable_body.add_joint(
            name="obj_t_joint_y", axis=[0, 1, 0], frictionloss=friction_loss, damping=damping,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
        )
        graspable_body.add_joint(
            name="obj_t_joint_z", axis=[0, 0, 1], frictionloss=friction_loss, damping=damping,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
        )

        graspable_body.add_joint(
            name="obj_r_joint_x", axis=[1, 0, 0], frictionloss=friction_loss/10, damping=damping,
            type=mujoco.mjtJoint.mjJNT_HINGE,
        )
        graspable_body.add_joint(
            name="obj_r_joint_y", axis=[0, 1, 0], frictionloss=friction_loss/10, damping=damping,
            type=mujoco.mjtJoint.mjJNT_HINGE,
        )
        graspable_body.add_joint(
            name="obj_r_joint_z", axis=[0, 0, 1], frictionloss=friction_loss/10, damping=damping,
            type=mujoco.mjtJoint.mjJNT_HINGE,
        )

    def get_obs_vector(self, obs_dict: Dict[str, Any]) -> tuple[NDArray, List[Tuple[str, int, str, Union[int, Tuple[int, ...]]]]]:
        sorted_keys = sorted(obs_dict.keys())
        obs_list = []
        metadata = []
        
        for key in sorted_keys:
            value = obs_dict[key]
            if isinstance(value, np.ndarray):
                flat_value = value.flatten()
                obs_list.append(flat_value)
                metadata.append((key, flat_value.size, "array", value.shape))
            elif isinstance(value, float):
                obs_list.append([value])
                metadata.append((key, 1, "float", 1))
        
        return np.hstack(obs_list), metadata

    def step(self, action: NDArray[np.float32]) -> tuple[NDArray, np.float32, bool, bool, dict[str, Any]]: # type: ignore
        
        truncated_action = np.clip(action, self.action_space.low, self.action_space.high)
        maped_action = self.convert_action_to_model(truncated_action)
      
        if self.kinematics_debug:
            self.do_simulation(maped_action, self.frame_skip)
            set_position_kinematics(self.data, self.hand_final_full_pose)
        else:
            self.do_simulation(maped_action, self.frame_skip)

        full_dict_obs = self._get_full_obs()
        reward, decomposed_reward = self.reward(full_dict_obs, action)
        reduced_obs, metadata = self.get_obs_vector(full_dict_obs)
        if self.render_mode == "human":
            self.render()
        
        debug_dict = {}
        debug_dict["state"] = self.get_state()
        debug_dict["full_obs"] = full_dict_obs
        debug_dict["decomposed_reward"] = decomposed_reward
        debug_dict["obs_metadata"] = metadata
        
        return reduced_obs, reward, False, False, debug_dict
    
    def set_state(self, qpos: NDArray[np.float64], qvel: Optional[NDArray[np.float64]] = None) -> None:
        """
        Set the state of the environment.
        """

        if qvel is None:
            number_pos = len(self.data.qpos)
            number_vel = len(self.data.qvel)
            qp = qpos[:number_pos]
            qv = qpos[number_pos:number_pos + number_vel]
        else:
            qp = qpos
            qv = qvel
    
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) ->  NDArray[np.float64]:
        state_vec = np.hstack([self.data.qpos, self.data.qvel])
        return state_vec



    def reset_mpc(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)
        ret_state_dict = {}
        ret_state_dict["qpos"] = self.data.qpos
        ret_state_dict["qvel"] = self.data.qvel
        res = np.hstack((self.data.qpos, self.data.qvel))
        return res

    def convert_action_to_model(self, action: np.ndarray) -> np.ndarray:
        # Overide the action space to match the mujoco model
        return action
    
    def reward(self, obs_dict: Dict[str, np.float32], acton: Optional[np.ndarray]=None) -> tuple[np.float32, dict]:
        weight_keypoints_error = self.reward_dict["distance_key_points"]
        weight_obj_error = self.reward_dict["obj_displacement"]
        weight_wirst_orient = self.reward_dict.get("diff_orient", 0)
        weight_obj_speed = self.reward_dict.get("obj_speed", 0)
        sum_error_dist = np.sum(obs_dict["distance_key_points_array"])

        r1 = -sum_error_dist * weight_keypoints_error
        r2 = -obs_dict["object_error"] * weight_obj_error
        r3 = -obs_dict["anglediff"] * weight_wirst_orient
        r4 = -obs_dict["obj_speed"] * weight_obj_speed
        reward = r1 + r2 + r3 + r4

        decompose = {
            "distance_key_points": r1,
            "obj_displacement": r2,
            "diff_orient": r3,
            "obj_speed": r4,
        }
        return reward, decompose
    
    def get_q_pos_hand(self) -> NDArray[np.float64]:
        q_poses = np.zeros(len(self.hand_final_full_pose.keys()))
        for num, key_i in enumerate(self.hand_final_full_pose.keys()):
            joint_id = self.data.model.joint(name=key_i).id
            qpos_i = self.data.qpos[joint_id]
            q_poses[num] = qpos_i
        return q_poses
    
    def _get_full_obs(self) -> Dict[str, Any]:
        """
        Get the observation of the environment.
        """
        qpos = self.data.qpos.ravel()
        key_bodies_dict_pose = get_key_bodies_pose(self.model, self.data)
        distances = {
            key: np.linalg.norm(np.array(key_bodies_dict_pose[key]) - np.array(self.key_pose_dict[key]))
            for key in self.key_pose_dict.keys()
        }

        tips_pose = np.zeros((len(self.tip_body_names), 3))
        for i, b_name in enumerate(self.tip_body_names):
            body_id = self.data.model.body(name=b_name).id
            body_centr_pose = self.data.xipos[body_id]
            tips_pose[i] = body_centr_pose

        body_palm_id = self.data.model.body(name=self.palm_name).id
        palm_pose = self.data.xipos[body_palm_id]
        act_joint_pose = self.data.actuator_length
         
        body_grasp_id = self.data.model.body(name="graspable_object").id
        object_pose = self.data.xipos[body_grasp_id]
        object_quat = self.data.xquat[body_grasp_id]
        current_control = self.data.ctrl


        distance_key_points_array = np.array(list(distances.values()))
        graspable_obj_name = "graspable_object"
        body_id = self.data.model.body(name=graspable_obj_name).id
        
        body_pose = self.data.xpos[body_id]
        object_erorr = np.linalg.norm(body_pose - self.obj_start_pos)

        x_rot = self.data.joint(name="WRJRx").qpos 
        y_rot = self.data.joint(name="WRJRy").qpos 
        z_rot = self.data.joint(name="WRJRz").qpos
        quaternion_real = euler.euler2quat(x_rot, y_rot, z_rot)
        
        x_rot_fin = self.hand_final_full_pose["WRJRx"]
        y_rot_fin = self.hand_final_full_pose["WRJRy"]
        z_rot_fin = self.hand_final_full_pose["WRJRz"]
        quaternion_final = euler.euler2quat(x_rot_fin, y_rot_fin, z_rot_fin)
        quaternion_conjugate = quaternions.qconjugate(quaternion_final)
        diff_quat = quaternions.qmult(quaternion_conjugate, quaternion_real)
        anglediff  = np.linalg.norm(np.array([1, 0, 0, 0]) - diff_quat)



        joint_id_x = self.data.model.joint(name="obj_t_joint_x").id
        joint_id_y = self.data.model.joint(name="obj_t_joint_y").id
        joint_id_z = self.data.model.joint(name="obj_t_joint_z").id
        obj_speed_x = self.data.qvel[joint_id_x]
        obj_speed_y = self.data.qvel[joint_id_y]
        obj_speed_z = self.data.qvel[joint_id_z]
        obj_speed = np.linalg.norm(np.array([obj_speed_x, obj_speed_y, obj_speed_z]))

        hand_q_poses = self.get_q_pos_hand()

        return  {
            "distance_key_points_array": distance_key_points_array,
            "obj_speed": obj_speed,
            "anglediff": anglediff,
            "object_error": object_erorr,
            "tips_pose": tips_pose,
            "palm_pose": palm_pose,
            "act_joint_pose": act_joint_pose,
            "object_pose": object_pose,
            "object_quat": object_quat,
            "current_control": current_control,
            "hand_q_poses": hand_q_poses,

        }


# fmt: on
