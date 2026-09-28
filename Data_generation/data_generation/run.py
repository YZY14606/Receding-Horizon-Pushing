import multiprocessing as mp
from . import env_creation  # noqa: registers Push_solve with Gymnasium
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from .record import RecordVideo
from .solve import solve


MP_SOLUTIONS = {
    "Push_solve": solve,
}
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="Push_solve", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="pointcloud", help="Observation mode used while generating data.")
    parser.add_argument("-n", "--num-traj", type=int, default=1, help="Number of trajectories to generate.")
    parser.add_argument("--obj-idx", type=int, default=2, help="Object index (integer). Default: 1",)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default= "cpu", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="Session name used for trajectory directories.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="Directory where optional videos are saved.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--demo_dir", type=str,
                        default="scene_data",
                        help="Path to original dataset directory")
    
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="none" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        num_envs = 1,
        reconfiguration_freq = 1,
        object_index = args.obj_idx,
        control_frequency = 30,
        scene_dir = args.demo_dir
    )
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")
    
    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    
    env = RecordVideo(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        video_fps=30,
        avoid_overwriting_video=True,
    )
    solve = MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(total=args.num_traj, desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    for _ in range(args.num_traj):
        try:
            res = solve(env, seed=seed, debug=False, vis=True if args.vis else False )
        except Exception as e:
            print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        if args.save_video:
            env.flush_video()
        env.advance_episode()
        pbar.update(1)
        pbar.set_postfix(
            dict(
                success_rate=np.mean(successes),
                failed_motion_plan_rate=failed_motion_plans / len(successes),
                avg_episode_length=(
                    np.mean(solution_episode_lengths)
                    if solution_episode_lengths
                    else 0.0
                ),
            )
        )
        seed += 1

    pbar.close()
    env.close()

def main(args):

    num_procs = min(args.num_procs, args.num_traj)
    base_count, remainder = divmod(args.num_traj, num_procs)

    proc_args = []
    start_seed = 0

    for proc_id in range(num_procs):
            worker_args = deepcopy(args)
            worker_args.num_procs = num_procs
            worker_args.num_traj = base_count + (proc_id < remainder)

            proc_args.append(
                (worker_args, proc_id, start_seed)
            )
            start_seed += worker_args.num_traj


    if num_procs == 1:
        _main(*proc_args[0])
    else:
        with mp.Pool(num_procs) as pool:
            pool.starmap(_main, proc_args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())

    
