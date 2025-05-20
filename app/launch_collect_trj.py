from icem_mpc.launch_colect import ConfigICEM, run_object_run


reward_dict = {
    "distance_key_points": 1.5,
    "obj_displacement": 2.0,
    "diff_orient": 1.0,
    "obj_speed": 1.0,
}

dummy_config_icem = ConfigICEM()
dummy_config_icem.horizon = 3
dummy_config_icem.mpc_steps = 3
dummy_config_icem.warmup_iters = 10
dummy_config_icem.online_iters = 10
dummy_config_icem.num_samples = 5

dummy_config_icem.num_elites = 3
dummy_config_icem.elites_keep_fraction = 0.01
dummy_config_icem.alpha = 0.003

dummy_config_icem.num_samples_after_reset = 5
dummy_config_icem.reset_penalty_thr = -0.8
dummy_config_icem.num_elites_after_reset = 3

run_object_run(
    reward_dict,
    dummy_config_icem,
    obj_name="core-bowl-a593e8863200fdb0664b3b9b23ddfcbc",
    pose_nums=[0, 1],
    folder="experts_traj_dummy",
    n_jobs=1
)
