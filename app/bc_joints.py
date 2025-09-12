from icem_mpc.bc import BC
from icem_mpc.bc import train_behavior_cloning

import torch
import numpy as np
import os
import glob

dataset = []
hand_name = "shadow_dexee"
folder = "experts_traj_" + hand_name + "/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03"
filenames = [y for x in os.walk(folder) for y in glob.glob(os.path.join(x[0], '*.npz'))]

for file_name in filenames:
    load_file = np.load(file=file_name, allow_pickle=True, fix_imports=True)
    dataset.append(load_file)
model =  BC(18, 18)
# model.load_state_dict(torch.load('model_joints_weights.pth'))

observations = []
actions = []
for data in dataset:
    a = data['action_seq']
    o = data['full_observations']
    for i in range(len(a)):
        observations.append(o[i]['act_joint_pose'])
        actions.append(a[i])
        print('ssa', a[i], o[i]['act_joint_pose'])
observations = np.concat([observations])
actions = np.concat([actions])
print('act', actions)
print('obs', observations)
model = train_behavior_cloning(model, expert_observations=observations, expert_actions=actions)
torch.save(model.state_dict(), 'model_joints_weights.pth')
