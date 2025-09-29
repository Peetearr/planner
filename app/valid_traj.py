import os
import numpy as np
import glob

hand_name = "shadow_dexee"
folder = "experts_traj_" + hand_name + "/core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03/"
count = 0
filenames = glob.glob(os.path.join(folder, '*.npz'))
for file in filenames:
    data = np.load(file, allow_pickle=True)
    if data['costs_seq'][-1] > -1.6:
        count+=1
        basename = os.path.basename(file)
        print(file,' ok')
        source_file = folder + basename
        destination_file = folder + "valid_traj/" + basename
        os.system(f"cp -r '{source_file}' '{destination_file}'")
print('Количество успешных траекторий:', count)