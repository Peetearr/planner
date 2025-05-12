from dataclasses import dataclass
import itertools
import os
import time
from typing import Optional
import mujoco
import numpy as np
from config_crearor import get_tabale_top_start_pos, prepare_env_config
from pytorch_icem import iCEM
from torch import Tensor

from reach_pose_env import ReachPoseEnv, ReachPoseEnvConfig
import gymnasium as gym
from functools import partial
import matplotlib.pyplot as plt
from launch_colect import ConfigICEM, run_icem


def run_for_one_obj(object_folder: str, hand_start_poses: list[dict]):
    filenames = [f for f in os.listdir(object_folder) if os.path.isfile(os.path.join(object_folder, f))]

    return filenames
   
    
reward_dict = {
    "distance_key_points": 1.5,
    "obj_displacement": 10.0,
    "diff_orient": 3.0,
}

config_icem = ConfigICEM()
