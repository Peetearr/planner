from icem_mpc.launch_colect import ConfigICEM, run_object_run


reward_dict = {
    "distance_key_points": 1.5,
    "obj_displacement": 2.0,
    "diff_orient": 1.0,
    "obj_speed": 1.0,
}

dummy_config_icem = ConfigICEM()
dummy_config_icem.horizon = 10
dummy_config_icem.mpc_steps = 30
dummy_config_icem.warmup_iters = 120
dummy_config_icem.online_iters = 50
dummy_config_icem.num_samples = 30

dummy_config_icem.num_elites = 10
dummy_config_icem.elites_keep_fraction = 0.5
dummy_config_icem.alpha = 0.003

dummy_config_icem.num_samples_after_reset = 100
dummy_config_icem.reset_penalty_thr = -0.8
dummy_config_icem.num_elites_after_reset = 60

hand_name="dexee"
run_object_run(
    reward_dict,
    dummy_config_icem,
    obj_name="core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03",
    pose_nums=[6],
    folder="experts_traj_" + hand_name,
    n_jobs=4
)
