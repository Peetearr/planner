from copy import deepcopy
import time
import mujoco
import mujoco.viewer
import numpy as np
import transforms3d.euler as euler
from transforms3d import affines
from numpy.typing import NDArray

from icem_mpc.load_complex_obj import add_graspable_body, add_meshes_from_folder

Z_QUAT_0 = euler.euler2quat(np.deg2rad(5), np.deg2rad(0), np.deg2rad(10))


default_mapping = {
    "WRJTx": "WRJTx",
    "WRJTy": "WRJTy",
    "WRJTz": "WRJTz",
    "WRJRx": "WRJRx",
    "WRJRy": "WRJRy",
    "WRJRz": "WRJRz",
    "F0_J0": "F0_J0",
    "F0_J1": "F0_J1",
    "F0_J2": "F0_J2", 
    "F0_J3": "F0_J3", 
    "F1_J0": "F1_J0", 
    "F1_J1": "F1_J1",    
    "F1_J2": "F1_J2", 
    "F1_J3": "F1_J3", 
    "F2_J0": "F2_J0", 
    "F2_J1": "F2_J1", 
    "F2_J2": "F2_J2", 
    "F2_J3": "F2_J3"
}

def quintic_func(q0, qf, T, qd0=0, qdf=0):
    "Cite code from https://github.com/petercorke/robotics-toolbox-python"

    """
    Quintic scalar polynomial as a function

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param T: trajectory time
    :type T: float
    :param qd0: initial velocity, defaults to 0
    :type q0: float, optional
    :param qdf: final velocity, defaults to 0
    :type q0: float, optional
    :return: polynomial function :math:`f: t \mapsto (q(t), \dot{q}(t), \ddot{q}(t))`
    :rtype: callable

    Returns a function which computes the specific quintic polynomial, and its
    derivatives, as described by the parameters.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import quintic_func
        >>> f = quintic_func(1, 2, 5)
        >>> f(0)
        >>> f(5)
        >>> f(2.5)

    :seealso: :func:`quintic` :func:`trapezoidal_func`
    """

    # solve for the polynomial coefficients using least squares
    # fmt: off
    X = [
        [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
        [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
        [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
        [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
        [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
        [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
    ]
    # fmt: on
    coeffs, resid, rank, s = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
    )

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )


# -------------------------------------------------------------------------- #


def create_quintic_traj_function(q0: NDArray, qf: NDArray, t_final: float):
    def traj_function(time: float) -> NDArray:
        # for q0_i, qf_i in zip(q0, qf):
        #  q, dq, ddq = quintic_func(q0_i, qf_i, t_final)(time)
 
        return np.array([quintic_func(q0[i], qf[i], t_final)(time)  for i in range(len(q0))])
    return traj_function


def get_key_bodies_pose(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> dict[str, NDArray]:
    body_names = get_key_bodies_shadow_names(mj_model)

    body_pos_dict = {}
    for b_name in body_names:
        body_id = mj_data.model.body(name=b_name).id
        body_centr_pose = mj_data.xipos[body_id]
        body_pos_dict[b_name] = body_centr_pose
    return body_pos_dict


def set_position(mj_data: mujoco.MjData, qpos: dict[str, float], maping: dict[str, str] = None):
    if maping is None:
        for key, value in qpos.items():
            mj_data.actuator(key).ctrl = value
    else:
        for key, value in qpos.items():
            mj_data.actuator(maping[key]).ctrl = value


def set_position_kinematics(mj_data: mujoco.MjData, qpos: dict[str, float], maping: dict[str, str] = None):
    if maping is None:
        for key, value in qpos.items():
            qpos_id = mj_data.model.joint(name=key).qposadr
            mj_data.qpos[qpos_id] = value

    else:
        raise NotImplementedError("Mapping is not implemented for kinematics")


def get_key_bodies_shadow_names(composite_model):

    bodies_names = []
    for i in range(composite_model.nbody):
        bodies_names.append(composite_model.body(i).name)
    bodies_names = [name for name in bodies_names if  "distal" in name or "palm" in name ]
    return bodies_names


def transform_wirst_pos_to_obj(pos_obj: NDArray, quat_obj: NDArray, pos_hand: NDArray, quat_hand: NDArray) -> tuple[NDArray, NDArray]:

    rotation_matrix_obj = euler.quat2mat(quat_obj)

    homogeneous_matrix_obj = affines.compose(T=pos_obj, R=rotation_matrix_obj, Z=np.ones(3))

    rotation_matrix_hand = euler.quat2mat(quat_hand)
    # Create a homogeneous transformation matrix
    homogeneous_matrix_hand = affines.compose(T=pos_hand, R=rotation_matrix_hand, Z=np.ones(3))

    transformed_pos = homogeneous_matrix_obj.dot(homogeneous_matrix_hand)
    T, R, _, _ = affines.decompose(transformed_pos)
    return T, R



def add_body_key_points(spec_mujoco, key_pose_dict):
    for pose_name, pose in key_pose_dict.items():
        spec_mujoco.worldbody.add_geom(name=pose_name + "ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=[1, 1, 0, 0.25], size=[0.005, 0.005, 0.1], pos=pose)


def get_final_bodies_pose(final_position: dict[str, float], hand_model_path: str):
 
    model_for_pose = mujoco.MjModel.from_xml_path(hand_model_path)
    data_for_pose = mujoco.MjData(model_for_pose)

    set_position_kinematics(data_for_pose, final_position)
    mujoco.mj_kinematics(model_for_pose, data_for_pose)

    key_bodies_pose = get_key_bodies_pose(model_for_pose, data_for_pose)
    return key_bodies_pose

def main():
    POSE_NUM = 10
    obj_name = "sem-Plate-9969f6178dcd67101c75d484f9069623"
    pos_path_name = "final_positions/" + obj_name + ".npy"
    mesh_path = "mjcf/model_dexgraspnet/meshes/objs/" + obj_name + "/coacd"

    core_mug = np.load(pos_path_name, allow_pickle=True)

    spec = mujoco.MjSpec.from_file("./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml")

    # body.add_joint(name="free", type=mujoco.mjtJoint.mjJNT_FREE)
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

    combined_mesh, mesh_names = add_meshes_from_folder(
        spec,
        mesh_path,
        prefix="obj_",
        scale=[
            core_mug[POSE_NUM]["scale"],
            core_mug[POSE_NUM]["scale"],
            core_mug[POSE_NUM]["scale"],
        ],
    )
    graspable_body = add_graspable_body(spec, combined_mesh, mesh_names, density=1000 * 0.14)
    for gg in graspable_body.geoms:
        gg.friction = [0.8, 0.009, 0.0001]

    graspable_body.gravcomp = 0.9
    graspable_body.add_joint(
        name="obj_t_joint_x",
        axis=[1, 0, 0],
        frictionloss=4,
        damping=0.5,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
    )
    graspable_body.add_joint(
        name="obj_t_joint_y",
        axis=[0, 1, 0],
        frictionloss=4,
        damping=0.5,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
    )
    graspable_body.add_joint(
        name="obj_t_joint_z",
        axis=[0, 0, 1],
        frictionloss=3,
        damping=0.5,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
    )

    graspable_body.add_joint(
        name="obj_r_joint_x",
        axis=[1, 0, 0],
        frictionloss=0.1,
        damping=0.1,
        type=mujoco.mjtJoint.mjJNT_HINGE,
    )
    graspable_body.add_joint(
        name="obj_r_joint_y",
        axis=[0, 1, 0],
        frictionloss=0.1,
        damping=0.1,
        type=mujoco.mjtJoint.mjJNT_HINGE,
    )
    graspable_body.add_joint(
        name="obj_r_joint_z",
        axis=[0, 0, 1],
        frictionloss=0.1,
        damping=0.1,
        type=mujoco.mjtJoint.mjJNT_HINGE,
    )
    # spec.body("graspable_object").pos = self.obj_start_pos
    # spec.find_body("graspable_object").pos = np.array([0, 0, 0])
    composite_model = spec.compile()
    composite_data = mujoco.MjData(composite_model)

    # Print all actuator names
    # print("Actuator names:")
    # for i in range(model.nu):
    #     print(f"Actuator {i}: {model.actuator(i).name}")

    final_position = deepcopy(core_mug[POSE_NUM]["qpos"])
    # Filter final position to only include wrist coordinates

    wrist_names = [
        "WRJTx",
        "WRJTy",
        "WRJTz",
        "WRJRx",
        "WRJRy",
        "WRJRz",
    ]

    final_position_wirst = {k: v for k, v in final_position.items() if k in wrist_names}

    obj_quat = euler.euler2quat(np.deg2rad(90), np.deg2rad(180), np.deg2rad(180))
    obj_pos = [0.0, 0, 0.4]

    wirst_pos = np.array(
        [
            final_position_wirst["WRJTx"],
            final_position_wirst["WRJTy"],
            final_position_wirst["WRJTz"],
        ]
    )
    wirst_quat = euler.euler2quat(
        final_position_wirst["WRJRx"],
        final_position_wirst["WRJRy"],
        final_position_wirst["WRJRz"],
    )

    coca = transform_wirst_pos_to_obj(obj_pos, obj_quat, wirst_pos, wirst_quat)

    euler_ang = euler.mat2euler(coca[1])

    new_pos = {
        "WRJRx": euler_ang[0],
        "WRJRy": euler_ang[1],
        "WRJRz": euler_ang[2],
        "WRJTx": coca[0][0],
        "WRJTy": coca[0][1],
        "WRJTz": coca[0][2],
    }
    # set_position(composite_data, new_pos, default_mapping)

    translation_names = ["WRJTx", "WRJTy", "WRJTz"]
    rot_names = ["WRJRz", "WRJRy", "WRJRx"]

    for key in new_pos.keys():
        final_position[key] = new_pos[key]

    joint_names = [
        "robot0:FFJ3",
        "robot0:FFJ2",
        "robot0:FFJ1",
        "robot0:FFJ0",
        "robot0:MFJ3",
        "robot0:MFJ2",
        "robot0:MFJ1",
        "robot0:MFJ0",
        "robot0:RFJ3",
        "robot0:RFJ2",
        "robot0:RFJ1",
        "robot0:RFJ0",
        "robot0:LFJ4",
        "robot0:LFJ3",
        "robot0:LFJ2",
        "robot0:LFJ1",
        "robot0:LFJ0",
        "robot0:THJ4",
        "robot0:THJ3",
        "robot0:THJ2",
        "robot0:THJ1",
        "robot0:THJ0",
    ]

    # composite_data.qpos[0] = -0.2
    # composite_data.qpos[1] = 0.2
    # composite_data.qpos[2] = 0.2

    composite_model.body("graspable_object").pos = obj_pos
    composite_model.body("graspable_object").quat = obj_quat
    num_actuators = composite_model.nu
    print(f"Number of actuators: {num_actuators}")
    counter = 0
    model_for_pose_path = "./mjcf/model_dexgraspnet/shadow_hand_wrist_free_special_path.xml"
    
   
    
    fin_pose = get_final_bodies_pose(final_position, model_for_pose_path)
    #set_position_kinematics(composite_data, final_position)
    
    viewer = mujoco.viewer.launch_passive(composite_model, composite_data)
    while True:
        counter += 1
        step_start = time.time()

        if counter == 100:
            print("Change")
            set_position(composite_data, final_position, default_mapping)

        if counter == 1000:
            print("Enabale gravity")

        mujoco.mj_step(composite_model, composite_data)
        viewer.sync()
        time_until_next_step = composite_model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)



if __name__ == "__main__":
    main()
