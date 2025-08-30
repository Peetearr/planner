from icem_mpc.launch_colect import ConfigICEM, run_object_run


reward_dict = {
    "distance_key_points": 1.5,
    "obj_displacement": 2.0,
    "diff_orient": 1.0,
    "obj_speed": 1.0,
}
config_icem = ConfigICEM()
config_icem.horizon = 6
config_icem.mpc_steps = 30
config_icem.warmup_iters = 100
config_icem.online_iters = 50
config_icem.num_samples = 35

config_icem.num_elites = 10
config_icem.elites_keep_fraction = 0.5
config_icem.alpha = 0.003

config_icem.num_samples_after_reset = 60
config_icem.reset_penalty_thr = -0.5
config_icem.num_elites_after_reset = 20

hand_name="shadow_dexee"
run_object_run(
    reward_dict,
    config_icem,
    obj_name="core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03",
    pose_nums=[12,13,14,15,16,17,18,19],
    folder="experts_traj_" + hand_name,
    n_jobs=8
)
