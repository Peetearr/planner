import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path('mjcf/model_dexgraspnet/shadow_dexee.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  while viewer.is_running():
    
    mujoco.mj_step(m, d)
    time.sleep(.02)
    viewer.sync()