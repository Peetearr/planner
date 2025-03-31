import numpy as np
import xml.etree.ElementTree as ET
from hand_imitation.env.environments.base import MujocoEnv

from hand_imitation.env.models import TableArena
from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements
from hand_imitation.env.utils.random import np_random
from hand_imitation.env.models.objects import YCB_SIZE, YCB_ORIENTATION
import transforms3d
import trimesh

class SHAPENETRelocate(MujocoEnv):
    def __init__(self, has_renderer, render_gpu_device_id=-1, object_name="0001", category="mug", object_scale=1, randomness_scale=1,
                 **kwargs):
        self.np_random = None
        self.seed()
        self.object_name = object_name
        self.category = category
        self.object_scale = object_scale
        self.randomness_scale = randomness_scale
        self.__additional_kwargs = kwargs
        #"camera" names = ('frontview', 'agentview', 'birdview', 'sideview')
        super().__init__(has_renderer=has_renderer, has_offscreen_renderer=False, render_camera=None,
                         render_gpu_device_id=render_gpu_device_id, control_freq=100, horizon=100, ignore_done=True,
                         hard_reset=False)

        # Setup action range
        self.act_mid = np.mean(self.mjpy_model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.mjpy_model.actuator_ctrlrange[:, 1] - self.mjpy_model.actuator_ctrlrange[:, 0])
        #self.target_robot_pos=target_robot_pos
        
    def _pre_action(self, action, policy_step=False):
        action = np.clip(action, -1.0, 1.0)
        action = self.act_mid + action * self.act_rng  # mean center and scale
        self.sim.data.ctrl[:] = action
        # state=self.sim.get_state()
        # state[1][:30]=action
        # self.sim.set_state(state)
        # self.sim.step()

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        self.sim.set_state(self.sim_state_initial)
        self.sim.forward()
        self.sim.data.qpos[30] = self.np_random.uniform(low=0.15, high=0.15) * self.randomness_scale - 0.1
        self.sim.data.qpos[31] = self.np_random.uniform(low=-0.15, high=0.15) * self.randomness_scale
        self.mjpy_model.body_pos[self.target_object_bid, 0] = self.np_random.uniform(low=-0.30, high=0.10)
        self.mjpy_model.body_pos[self.target_object_bid, 1] = self.np_random.uniform(low=-0.15, high=0.15)
        self.mjpy_model.body_pos[self.target_object_bid, 2] = self.np_random.uniform(low=0.15, high=0.25)

    def _get_observations(self):
        qp = self.data.qpos.ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.body_xpos[self.target_object_bid].ravel()
        return np.concatenate([qp[:30], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos])
    
    def set_target_robot_pos(self,target_robot_pos):
        self.target_robot_pos=target_robot_pos
    
    def set_target_hand_joints(self,target_hand_joints,obj_pose):
        self.target_hand_joints=target_hand_joints
        self.initial_obj_pose=obj_pose
    
    def get_traj(self,obj_pose_path,hand_joints_path):
        self.obj_pose_seq=np.load(obj_pose_path,allow_pickle=True)
        self.hand_joints_seq=np.load(hand_joints_path,allow_pickle=True)
    
    def reward(self, action, t):
        # return 0
        affordance_v2=True
        baseline=False

        



        
        if affordance_v2:
            target_hand_joints=self.target_hand_joints
            target_obj_pos=self.initial_obj_pose[:3]
            #target_obj_quat=self.initial_obj_pose[3:]

            target_hand_palm_tips=target_hand_joints[[0,17,18,19,20,16]].ravel()
            

            #target_hand_tips=target_hand_joints[[8,12,16,20,4]].ravel()
            

            obj_pos=self.data.body_xpos[-2]#[1:-1]
            #obj_quat=self.data.body_xquat[-2]
            
            robot_tips=self.data.site_xpos[[8,12,16,21,25]].ravel()
            palm_pos = self.data.site_xpos[self.S_grasp_sid]


            #need=[2,3,7,11,15,20,25]
            reward=0
            

            

            
            
            reward = -0.1 * np.linalg.norm(target_hand_palm_tips - np.concatenate([palm_pos,robot_tips]))



            reward = reward - np.linalg.norm(target_obj_pos - obj_pos)

        elif baseline:
            
            

            
            palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()

            
            obj_pos = self.data.body_xpos[self.obj_bid].ravel()


            target_obj_pos=self.initial_obj_pose[:3].ravel()

            reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)

            #reward = -0.1 * np.linalg.norm(target_hand_tips - robot_tips)

            reward = reward - np.linalg.norm(target_obj_pos - obj_pos)
            


            
        return reward

    def _setup_references(self):
        self.target_object_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id(self.object_body_name)

    def _load_model(self):
        arena = TableArena(table_full_size=(1.2, 1.2, 0.05), table_friction=(1, 0.5, 0.01), table_offset=(0, 0, 1.0),
                           bottom_pos=(0, 0, -1), has_legs=True)
        xml_file = xml_path_completion("adroit/adroit_relocate.xml")
        robot = MujocoXML(xml_file)
        mesh_list = find_elements(robot.worldbody[0], tags="geom", return_first=False)
        robot_geom_names = [geom.get("name", "") for geom in mesh_list]
        self.robot_geom_names = [name for name in robot_geom_names if len(name) > 0]

        # Add YCB Object for relocation task
        obj_now=trimesh.load('./hand_imitation/env/models/assets/shapenet_'+self.category+'/visual/'+self.object_name+'/model_transform_scaled.stl')

        object_size = tuple(abs(obj_now.bounding_box.bounds[0]-obj_now.bounding_box.bounds[1]))

        # Parse args for YCB object geom
        additional_kwargs = self.__additional_kwargs.copy()
        for key, value in additional_kwargs.items():
            if isinstance(value, (np.ndarray, tuple, list, float, int)):
                additional_kwargs[key] = array_to_string(value)
        if "condim" not in additional_kwargs:
            additional_kwargs["condim"] = "4"
        if "margin" not in additional_kwargs:
            additional_kwargs["margin"] = "0.003"

        # Add ycb object into the arena and cache its body name
        arena.add_shapenet_object(self.category, self.object_name, pos=[0, 0, object_size[2] / 2], quat=YCB_ORIENTATION['mug'],
                             free=True, density=1000, idn=0, scale=self.object_scale, **additional_kwargs)
        self.object_body_name = arena.objects[0].body_name
        object_geom = find_elements(arena.objects[0].body, "geom", return_first=False)
        self.body_geom_names = [geom.get("name", "") for geom in object_geom if
                                geom.get("name", "").startswith("collision")]
        self.body_geom_names.append("table_collision")
        # Add target visualization for relocation task
        target_position = np.array(object_size) / 2 + np.array([0.05, 0.05, 0.12])
        object_target = ET.Element("body", name="target", pos=array_to_string(target_position),
                                   quat=array_to_string(YCB_ORIENTATION['mug']))
        target_geom = ET.Element("geom", type="mesh", mesh=f"{self.object_name}_visual_mesh", contype="0",
                                 conaffinity="0", rgba="0 1 0 0") #################0.125
        object_target.append(target_geom)
        arena.worldbody.append(object_target)

        # Merge robot xml with table arena
        robot.merge(arena, merge_body="default")
        self.model = robot
        self.model.save_model("relocate_temp.xml")

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    @property
    def action_spec(self):
        high = np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 1])
        low = -1.0 * np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 0])
        return low, high

    def set_state(self, qpos, qvel):
        import mujoco_py
        assert qpos.shape == (self.mjpy_model.nq,) and qvel.shape == (self.mjpy_model.nv,)

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = self.sim.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.sim.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.sim.data.site_xpos[self.target_object_bid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos, qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.mjpy_model.body_pos[self.obj_bid] = obj_pos
        self.mjpy_model.site_pos[self.target_object_bid] = target_pos
        self.sim.forward()

    @property
    def spec(self):
        this_spec = Spec(self._get_observations().shape[0], self.action_spec[0].shape[0])
        return this_spec

    def set_seed(self, seed=None):
        return self.seed(seed)


class Spec:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
