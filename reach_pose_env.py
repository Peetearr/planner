from copy import deepcopy
import random
from typing import Dict, Optional, Tuple
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import transforms3d.euler as euler
from transforms3d import affines
from transforms3d import quaternions
from grasp_env_utils import (
    default_mapping,
    get_final_bodies_pose,
    get_key_bodies_pose,
    set_position_kinematics,
    transform_wirst_pos_to_obj,
)
from load_complex_obj import add_graspable_body, add_meshes_from_folder
from numpy.typing import NDArray
from dataclasses import dataclass, field
from pytorch_icem import iCEM, icem
from joblib import Parallel, delayed
from torch import Tensor


@dataclass
class ReachPoseEnvConfig:
    model_path_hand: str
    obj_mesh_path: str
    hand_final_full_pose: Dict[str, float]
    obj_start_pos: np.ndarray
    obj_start_quat: np.ndarray
    obj_scale: float = 1.0
    frame_skip: int = 5
    hand_starting_pose: Optional[Dict[str, float]] = None


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
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.key_pose_dict = key_pose_dict
        self.kinematics_debug = False

        self.action_space = self.init_action_space(config.model_path_hand)

        # TODO: Define observation space properly
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        super().__init__(
            model_path=config.model_path_hand,
            frame_skip=config.frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )

        self.reset()

    def add_body_key_points(self):
        for pose_name, pose in self.key_pose_dict.items():
            self.spec_mujoco.worldbody.add_geom(
                name=pose_name + "_ball",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=[1, 1, 0.1, 0.5],
                size=[0.003, 0.003, 0.003],
                pos=pose,
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)
        return obs, info

    def reset_model(self) -> np.ndarray:
        """
        Reset the state of the environment to an initial state.
        """

        # self.data.qpos[:] = self.init_qpos
        # self.data.qvel[:] = self.init_qvel
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
        # sdfs = sorted()
        spec_mujoco = mujoco.MjSpec.from_file(self.fullpath)
        self.spec_mujoco = spec_mujoco

        self.add_body_key_points()

        # for key, value in self.hand_final_full_pose.items():
        #     try:
        #         res = spec_mujoco.find_actuator(key)
        #         if res is None:
        #             raise Exception("Actuator not found")
        #     except Exception as e:
        #         print(f"Actuator {key} not found")
        #         raise Exception("The final position and hand do not match")

        self.simultion_settings(spec_mujoco)
        self.add_graspable_object_spec(spec_mujoco)

        composite_model = spec_mujoco.compile()
        composite_data = mujoco.MjData(composite_model)

        composite_model.vis.global_.offwidth = self.width
        composite_model.vis.global_.offheight = self.height

        return composite_model, composite_data

    def simultion_settings(self, spec: mujoco.MjSpec):
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT  # type: ignore
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
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

        graspable_body.gravcomp = 1
        graspable_body.add_joint(
            name="obj_t_joint_x", axis=[1, 0, 0], frictionloss=4, damping=0.5,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
        )
        graspable_body.add_joint(
            name="obj_t_joint_y", axis=[0, 1, 0], frictionloss=4, damping=0.5,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
        )
        graspable_body.add_joint(
            name="obj_t_joint_z", axis=[0, 0, 1], frictionloss=3, damping=0.5,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
        )

        graspable_body.add_joint(
            name="obj_r_joint_x", axis=[1, 0, 0], frictionloss=0.1, damping=0.1,
            type=mujoco.mjtJoint.mjJNT_HINGE,
        )
        graspable_body.add_joint(
            name="obj_r_joint_y", axis=[0, 1, 0], frictionloss=0.1, damping=0.1,
            type=mujoco.mjtJoint.mjJNT_HINGE,
        )
        graspable_body.add_joint(
            name="obj_r_joint_z", axis=[0, 0, 1], frictionloss=0.1, damping=0.1,
            type=mujoco.mjtJoint.mjJNT_HINGE,
        )


    def step(self, action: NDArray[np.float32]) -> Tuple[NDArray, np.float32, bool, bool, Dict[str, np.float64]]:
        maped_action = self.map_action_to_model(action)
        copied_fin_pose = deepcopy(self.hand_final_full_pose)
        
        distance_key_points_array, object_erorr, diff_orient = self._get_obs()
        if self.kinematics_debug:
            mujoco.mj_kinematics(self.model, self.data)
            
            for key in self.hand_final_full_pose.keys():
                copied_fin_pose[key] += np.random.normal(0, 0.000001)
            set_position_kinematics(self.data, copied_fin_pose)
            state_1 = np.hstack((self.data.qpos, self.data.qvel))
            cosik = self.cost_mpc(state_1, 0)
            #print(cosik)
        else:
            self.do_simulation(maped_action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        
        reward = np.float32(0.0) 
        obs = np.hstack([distance_key_points_array, object_erorr])
        return obs, reward, False, False, {}
    
    def step_mpc(self, state_dict, action): 
        number_pos = len(self.data.qpos)
        number_vel = len(self.data.qvel)
        qp = state_dict[:number_pos]
        qv = state_dict[number_pos:number_pos + number_vel]
        self.set_state(qp, qv)
        self.do_simulation(action,  self.frame_skip)
        
        if self.render_mode == "human":
            self.render()

        res = np.hstack((self.data.qpos, self.data.qvel))
        return res
    
    def parralel_step_mpc(self, state_vector, action_vector):
        # res = Parallel(n_jobs=10, return_as="list")(
        # delayed(return_big_object)(i) for i in range(150))
        res_states = []
        for state, act in zip(state_vector, action_vector):
            state = self.step_mpc(state, act)
            res_states.append(state)

        tens=Tensor(res_states)
        return tens
    
    def cost_mpc(self, state_dict, action):
        number_pos = len(self.data.qpos)
        number_vel = len(self.data.qvel)
        qp = state_dict[:number_pos]
        qv = state_dict[number_pos:number_pos + number_vel]
        self.set_state(qp, qv)
        distance_key_points_array, object_erorr, diff_orient = self._get_obs()
        cost = -self.reward(distance_key_points_array, object_erorr, diff_orient)
        return cost
    
    def cost_mpc_vec(self, sate_vec, action_vec):
        number_pos = len(self.data.qpos)
        number_vel = len(self.data.qvel)
        vec_cost = []
        for state in sate_vec:
            qp = state[:number_pos]
            qv = state[number_pos:number_pos + number_vel]
            self.set_state(qp, qv)
            distance_key_points_array, object_erorr, diff_orient = self._get_obs()
            cost = -self.reward(distance_key_points_array, object_erorr, diff_orient)
            vec_cost.append(cost)
        return Tensor(cost)
    
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
 
    def map_action_to_model(self, action: np.ndarray) -> np.ndarray:
        # Overide the action space to match the mujoco model
        return action
    
    def reward(self, key_points_distance : NDArray[np.float32], obj_displacmnet: np.float32, diff_orient: np.float32) -> np.float32:
        weight_keypoints_error = 0.5
        mean_error_dist = np.sum(key_points_distance)
        #print(diff_orient)
        weight_obj_error = 1
        weight_wirst_orient = 1
        reward = -mean_error_dist*weight_keypoints_error - obj_displacmnet*weight_obj_error - diff_orient*weight_wirst_orient
        return reward
    
    

    def _get_obs(self) -> Tuple[np.ndarray, np.float32, np.float32]:
        """
        Get the observation of the environment.
        """
        qpos = self.data.qpos.ravel()
        key_bodies_dict_pose = get_key_bodies_pose(self.model, self.data)
        distances = {
            key: np.linalg.norm(np.array(key_bodies_dict_pose[key]) - np.array(self.key_pose_dict[key]))
            for key in self.key_pose_dict.keys()
        }
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

        return  distance_key_points_array, object_erorr, anglediff
# fmt: on
def debug_env():

    POSE_NUM = 3
    obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
    pos_path_name = "final_positions/" + obj_name + ".npy"
    mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"
    model_path_hand = "./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"


    obj_start_pos = np.array([0, -0.1, 0])
    obj_start_quat = euler.euler2quat(np.deg2rad(45), np.deg2rad(0), np.deg2rad(0))

    core_mug = np.load(pos_path_name, allow_pickle=True)
    qpos_hand = core_mug[POSE_NUM]["qpos"]

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

    config = ReachPoseEnvConfig(
        hand_final_full_pose=qpos_hand,
        model_path_hand=model_path_hand,
        obj_mesh_path=mesh_path,
        obj_start_pos=obj_start_pos,
        obj_start_quat=obj_start_quat,
        obj_scale=core_mug[POSE_NUM]["scale"],
        frame_skip = 2
    )

    reacher = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human")
    reacher.kinematics_debug = True
    for _ in range(5000):
        bingo_bongo = reacher.action_space.sample()
        # bingo_bongo[0:6] = [0, 0, 0, 0, 0, 0]
        for _ in range(100):
            #bingo_bongo[6] = 1
            bingo_action = np.zeros(reacher.action_space.shape)
            id_action_1 = reacher.data.actuator(name="WRJTy").id
            #bingo_action[id_action_1] = 0

            id_action_2 = reacher.data.actuator(name="WRJTx").id
            #bingo_action[id_action_2] = 4
            reacher.step(bingo_action)

        reacher.reset()

def run_mpc():
    POSE_NUM = 3
    obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
    pos_path_name = "final_positions/" + obj_name + ".npy"
    mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"
    model_path_hand = "./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"


    obj_start_pos = np.array([0.0, 0.0, 0])
    obj_start_quat = euler.euler2quat(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))

    core_mug = np.load(pos_path_name, allow_pickle=True)
    qpos_hand = core_mug[POSE_NUM]["qpos"]

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

    config = ReachPoseEnvConfig(
        hand_final_full_pose=qpos_hand,
        model_path_hand=model_path_hand,
        obj_mesh_path=mesh_path,
        obj_start_pos=obj_start_pos,
        obj_start_quat=obj_start_quat,
        obj_scale=core_mug[POSE_NUM]["scale"],
        frame_skip = 3
    )

    reacher = ReachPoseEnv(config, key_pose_dict=key_body_final_pos, render_mode="human")
    reacher_dynamic = ReachPoseEnv(config, key_pose_dict=key_body_final_pos)
    obs_mpc_state = reacher.reset_mpc()
    nu = reacher.action_space.shape[0]


    # create controller with chosen parameters
    def dynamics_mpc_wrapper(state_vec: Tensor, action_vec: Tensor):

        return reacher_dynamic.parralel_step_mpc(state_vec, action_vec)


    def cost_vec(sate_vec, action_vec):
        costs = []
        for trj in sate_vec:
            accum_cost_trj = 0
            for state in trj:
                step_cost = reacher_dynamic.cost_mpc(state, 0)
                accum_cost_trj += step_cost
            costs.append(accum_cost_trj)
            costs_tensor = Tensor(costs)
        return costs_tensor

    sigma = Tensor(reacher.action_space.high)
    ctrl = iCEM(
        dynamics=dynamics_mpc_wrapper,
        trajectory_cost=cost_vec,
        sigma=sigma,
        nx=68,
        nu=nu,
        warmup_iters=50,
        online_iters=50,
        num_samples=500,
        num_elites=50,
        horizon=5,
        device="cpu",
        alpha=0.3,
        noise_beta=10
    )

    # assuming you have a gym-like env
    obs_mpc_state = reacher.reset_mpc()
    for i in range(1000):
        action = ctrl.command(obs_mpc_state)
        obs_mpc_state = reacher.step_mpc(obs_mpc_state, action.cpu().numpy())
        #reacher.step(action.cpu().numpy())
        cost = reacher.cost_mpc(obs_mpc_state, 0)
        print(f"Cost : {cost}")


if __name__ == "__main__":
    run_mpc()