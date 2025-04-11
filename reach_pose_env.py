import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv

from load_complex_obj import add_graspable_body, add_meshes_from_folder

class ReachPoseEnv(MujocoEnv):
    def __init__(self, model_path_hand, obj_mesh_path, final_pose_dict: dict[str, float], object_scale = 1, frame_skip=5):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        super().__init__(model_path=model_path_hand, frame_skip=frame_skip, observation_space=observation_space)
        self.object_scale = object_scale
        self.final_pose_dict = final_pose_dict
        self.obj_mesh_path = obj_mesh_path
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(30,), dtype=np.float32)
        
        self.reset()
    
    def determine_dimentions(self):
        model = mujoco.MjModel.from_file(self.fullpath)
        model.nu 

    def _initialize_simulation(self):
        spec = mujoco.MjSpec.from_file(self.fullpath)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        combined_mesh, mesh_names = add_meshes_from_folder(spec, 
                                                           self.obj_mesh_path, 
                                                           prefix="obj_", 
                                                           scale=[self.object_scale, 
                                                                  self.object_scale, 
                                                                  self.object_scale])
        graspable_body = add_graspable_body(spec, combined_mesh, mesh_names, density=1000*0.14)

        for gg in graspable_body.geoms:
            gg.friction = [0.8, 0.009, 0.0001]


        graspable_body.gravcomp = 0.5
        graspable_body.add_joint(name = "obj_t_joint_x", axis = [1, 0, 0], frictionloss = 4 ,damping = 0.5, type = mujoco.mjtJoint.mjJNT_SLIDE)
        graspable_body.add_joint(name = "obj_t_joint_y", axis = [0, 1, 0], frictionloss = 4, damping = 0.5, type = mujoco.mjtJoint.mjJNT_SLIDE)
        graspable_body.add_joint(name = "obj_t_joint_z", axis = [0, 0, 1], frictionloss = 3, damping = 0.5, type = mujoco.mjtJoint.mjJNT_SLIDE)

        graspable_body.add_joint(name = "obj_r_joint_x", axis = [1, 0, 0], frictionloss = 0.1, damping = 0.1, type = mujoco.mjtJoint.mjJNT_HINGE)
        graspable_body.add_joint(name = "obj_r_joint_y", axis = [0, 1, 0], frictionloss = 0.1, damping = 0.1, type = mujoco.mjtJoint.mjJNT_HINGE)
        graspable_body.add_joint(name = "obj_r_joint_z", axis = [0, 0, 1], frictionloss = 0.1, damping = 0.1, type = mujoco.mjtJoint.mjJNT_HINGE)

        composite_model = spec.compile()
        composite_data = mujoco.MjData(composite_model)
        
        composite_model.vis.global_.offwidth = self.width
        composite_model.vis.global_.offheight = self.height
 
        return composite_model, composite_data
    
reacher = ReachPoseEnv(model_path_hand="./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml",
               obj_mesh_path="mjcf/model_dexgraspnet/meshes/objs/sem-Plate-9969f6178dcd67101c75d484f9069623/coacd",
               final_pose_dict={"obj_t_joint_x": 0.0, "obj_t_joint_y": 0.0, "obj_t_joint_z": 0.0,
                                "obj_r_joint_x": 0.0, "obj_r_joint_y": 0.0, "obj_r_joint_z": 0.0})
pass