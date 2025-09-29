import os
import numpy as np
import glob

hand_name = "shadow_dexee"
folder = "experts_traj_" + hand_name + "/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/"

filenames = glob.glob(os.path.join(folder, '*_mobile.pkl'))
for file in filenames:
    basename = os.path.basename(file)
    # os.remove(file)
    # os.rename(file, folder + basename.replace('_mobile.npz', '.npz'))

# filenames = glob.glob(os.path.join(folder, '*.pkl'))
# for file in filenames:
#     basename = os.path.basename(file)
#     os.rename(file, folder + basename.replace('.pkl', '_mobile.pkl'))