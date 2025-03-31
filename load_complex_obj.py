from copy import deepcopy
from pathlib import Path
import xml.etree
import gymnasium as gym
# import gymnasium_robotics
import time
import mujoco
import mujoco.viewer
import numpy as np

# from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
# from gymnasium_robotics.envs.adroit_hand import AdroitHandRelocateEnv 
import obj2mjcf.mjcf_builder
import transforms3d.euler as euler
import xml.etree.ElementTree as ET
from add_mass import update_urdf_with_mesh_properties
import os
import trimesh

def add_meshes_from_folder(mj_spec: mujoco.MjSpec, folder_path, prefix="_mesh", scale=[1, 1, 1]):
    """
    Add all meshes from a folder to the MJCF model.
    
    Args:
        model: The MJCF model to add meshes to.
        folder_path: The path to the folder containing the .obj files.
        prefix: Prefix to add to all names to avoid conflicts.
    """
 
    mesh_dir = mj_spec.meshdir
    modelfile_dirdir = mj_spec.modelfiledir
    # Get relative path from modelfiledir to folder_path
    rel_path = os.path.relpath(folder_path, modelfile_dirdir)
    # Set meshdir to this relative path
    # mj_spec.meshdir = rel_path
    mj_meshes_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".obj") and filename != "decomposed.obj":
            mesh_name = prefix + filename
            mesh_path = os.path.join(folder_path, filename)
            mesh_path2 = os.path.join(rel_path, filename)
            mj_mesh = mj_spec.add_mesh(file=mesh_path2, name=mesh_name, scale=scale)
            mj_meshes_names.append(mesh_name)
    convex_parts = [trimesh.load(os.path.join(folder_path,p)) for p in os.listdir(folder_path) if p.endswith(".obj")]
    combined_mesh = trimesh.util.concatenate(convex_parts)

    return combined_mesh, mj_meshes_names 

def add_graspable_body(mj_spec: mujoco.MjSpec, combined_mesh: trimesh.Trimesh, mj_meshes_names, obj_name: str = "graspable_object", density=1000):
    
    init_quat = euler.euler2quat(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))
    init_pos = [0.0, 0.0, 0.0]
    mass = 0.5
    
    body = mj_spec.worldbody.add_body(
        pos=init_pos,
        quat=init_quat,
        name=obj_name,
        mass = mass
    )
    for name in mj_meshes_names:
        body.add_geom(name=name + "_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        rgba=[1, 0, 0, 0.25],
        meshname=name,
        condim = 3)
    return body
 
             
def main():
    hand_path = "mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"
    urdf_path = "mjcf/model_dexgraspnet/obj_urdf/obj/core-mug-b7841572364fd9ce1249ffc39a0c3c0b/coacd/coacd.urdf"
    mesh_dir = "mjcf/model_dexgraspnet/meshes/objs/core-mug-b7841572364fd9ce1249ffc39a0c3c0b/coacd/"
    
    
    spec_hand = mujoco.MjSpec()
    

    combined_mesh, mesh_names = add_meshes_from_folder(spec_hand, "mjcf/model_dexgraspnet/meshes/objs/core-mug-b7841572364fd9ce1249ffc39a0c3c0b/coacd/", prefix="obj_", scale=[0.1, 0.1, 0.1])
    graspable_body = add_graspable_body(spec_hand, combined_mesh, mesh_names, density=1000*0.14)
    graspable_body.add_joint(name="free", type=mujoco.mjtJoint.mjJNT_FREE)
    composite_model = spec_hand.compile()
    composite_data = mujoco.MjData(composite_model)

    with mujoco.viewer.launch(composite_model, composite_data) as viewer:
        while True:
            step_start = time.time()
            mujoco.mj_step(composite_model, composite_data)
            viewer.sync()
            time_until_next_step = composite_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
#main()