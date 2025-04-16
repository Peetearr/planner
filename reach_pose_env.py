from typing import Dict, Optional, Tuple
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import transforms3d.euler as euler
from transforms3d import affines
from transforms3d import quaternions
from grasp_env_utils import default_mapping
from load_complex_obj import add_graspable_body, add_meshes_from_folder
from numpy.typing import NDArray


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


def transform_pos(
    pos_obj: np.ndarray,
    quat_obj: np.ndarray,
    pos_hand: np.ndarray,
    quat_hand: np.ndarray,
):

    rotation_matrix_obj = euler.quat2mat(quat_obj)

    homogeneous_matrix_obj = affines.compose(T=pos_obj, R=rotation_matrix_obj, Z=np.ones(3))

    rotation_matrix_hand = euler.quat2mat(quat_hand)
    # Create a homogeneous transformation matrix
    homogeneous_matrix_hand = affines.compose(T=pos_hand, R=rotation_matrix_hand, Z=np.ones(3))

    transformed_pos = homogeneous_matrix_obj.dot(homogeneous_matrix_hand)

    return affines.decompose(transformed_pos)


class ReachPoseEnv(MujocoEnv):
    def __init__(
        self,
        model_path_hand,
        obj_mesh_path,
        hand_final_full_pose: dict[str, float],
        obj_start_pos: np.ndarray,
        obj_start_quat: np.ndarray,
        obj_scale=1,
        frame_skip=5,
        hand_starting_pose: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        self.obj_scale = obj_scale
        self.obj_start_pos = obj_start_pos
        self.obj_start_quat = obj_start_quat
        self.obj_mesh_path = obj_mesh_path

        self.action_space = self.init_action_space(model_path_hand)
        self.hand_final_full_pose = hand_final_full_pose
        # TODO
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        super().__init__(
            model_path=model_path_hand,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs,
        )

        self.reset()

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

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
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

        for key, value in self.hand_final_full_pose.items():
            try:
                res = spec_mujoco.find_actuator(key)
                if res is None:
                    raise Exception("Actuator not found")
            except Exception as e:
                print(f"Actuator {key} not found")
                raise Exception("The final position and hand do not match")

        self.simultion_settings(spec_mujoco)
        self.add_graspable_object_spec(spec_mujoco)

        composite_model = spec_mujoco.compile()
        composite_data = mujoco.MjData(composite_model)

        composite_model.vis.global_.offwidth = self.width
        composite_model.vis.global_.offheight = self.height

        return composite_model, composite_data

    def simultion_settings(self, spec: mujoco.MjSpec):
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT  # type: ignore
        # spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        spec.option.timestep = 0.001

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
        
    def step(self, action: NDArray[np.float32]) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        maped_action = self.map_action_to_model(action)
        self.do_simulation(maped_action, self.frame_skip)
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        reward = 0.0 

        return obs, reward, False, False, {}
    
    def map_action_to_model(self, action: np.ndarray) -> np.ndarray:
        # Overide the action space to match the mujoco model
        return action
    
    def reward(self, state, action) -> float:
        return 0.0
    

    def _get_obs(self) -> np.ndarray:
        """
        Get the observation of the environment.
        """
        qpos = self.data.qpos.ravel()
 

        return np.concatenate(
            [qpos[:-6]]
        )
    # def set_env_state(self, state_dict):
    #     """
    #     Set the state which includes hand as well as objects and targets in the scene
    #     """
    #     # assert self._state_space.contains(
    #     #     state_dict
    #     # ), f"The state dictionary {state_dict} must be a member of {self._state_space}."
    #     qp = state_dict["qpos"]
    #     qv = state_dict["qvel"]

    #     self.model.body_pos[self.obj_body_id] = state_dict["obj_pos"]
    #     self.model.site_pos[self.target_obj_site_id] = state_dict["target_pos"]

    #     self.set_state(qp, qv)


# fmt: on
POSE_NUM = 10
obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
pos_path_name = "final_positions/" + obj_name + ".npy"
mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"

core_mug = np.load(pos_path_name, allow_pickle=True)
qpos_hand = core_mug[POSE_NUM]["qpos"]
qpos_shadow = convert_pose_dexgraspnet_to_mujoco(qpos_hand, default_mapping)


reacher = ReachPoseEnv(
    hand_final_full_pose=qpos_shadow,
    model_path_hand="./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml",
    obj_mesh_path="mjcf/model_dexgraspnet/meshes/objs/sem-Plate-9969f6178dcd67101c75d484f9069623/coacd",
    obj_start_pos=np.array([0.2, 0, 0]),
    obj_start_quat=np.array([0, 0, 0, 1]),
    obj_scale=core_mug[POSE_NUM]["scale"],
    render_mode="human",
)
for _ in range(5000):
    bingo_bongo = reacher.action_space.sample()
    # bingo_bongo[0:6] = [0, 0, 0, 0, 0, 0]
    for _ in range(100):
        reacher.step(bingo_bongo)

    reacher.reset()
