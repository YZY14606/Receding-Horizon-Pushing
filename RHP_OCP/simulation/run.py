import argparse
import multiprocessing as mp
import time
from copy import deepcopy
from pathlib import Path

import env_creation  # noqa: F401 - registers Push-v1 with Gymnasium
import gymnasium as gym
import numpy as np
from tqdm import tqdm

from artifact_layout import (
    ArtifactLayout,
    VALID_DIFFICULTIES,
    VALID_OBJECT_TYPES,
    build_worker_assignments,
)
from slove_push import solve as solve_push
from video_recorder import VideoRecorder

MP_SOLUTIONS = {
    "Push-v1": solve_push,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="Push-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument(
        "-o",
        "--obs-mode",
        type=str,
        default="pointcloud",
        help="Observation mode passed to the environment.",
    )
    parser.add_argument("-n", "--num-traj", type=int, default=1, help="Number of trajectories to generate.")
    parser.add_argument("--obj-idx", type=int, default=3, help="Object index (integer). Default: 1",)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default= "cpu", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="Root directory for recorded videos")
    parser.add_argument("--run-id", type=str, help="Shared identifier for one script/test.sh run")
    parser.add_argument("--num-procs", type=int, default=2, help="Number of CPU worker processes. Each worker writes to its own worker_NNN directory")
    parser.add_argument("--object_type", choices=sorted(VALID_OBJECT_TYPES), default="train", help="The object type")
    parser.add_argument("--scene_difficulty", choices=sorted(VALID_DIFFICULTIES), default="scene_01", help="The scene difficulty")
    parser.add_argument("--push-test-count", type=int, default=1,
    help="Maximum number of consecutive virtual push steps to validate.",)
    
    return parser.parse_args(args)


def _discard_failed_attempt(env, artifact_layout, traj_id):
    """Discard video frames and other artifacts from a failed attempt."""
    env.discard_video()
    artifact_layout.discard_episode_artifacts(traj_id)


def _main(
    args,
    artifact_layout: ArtifactLayout,
    proc_id: int = 0,
    start_seed: int = 0,
    seed_stride: int = 1,
) -> None:
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
        object_type = args.object_type,
        scene_difficulty = args.scene_difficulty
    )

    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")
    
    env = VideoRecorder(
        env,
        output_dir=str(artifact_layout.video_dir),
        enabled=args.save_video,
        fps=30,
    )
    solve = MP_SOLUTIONS[env_id]
    print(
        f"Motion Planning Running on {env_id}; "
        f"artifact task={artifact_layout.task_relative_dir}"
    )
    pbar = tqdm(
        range(args.num_traj),
        desc=f"worker_{proc_id:03d}",
    )
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    attempt_count = 0
    passed = 0
    retry_count = 0
    MAX_RETRY_COUNT = 5

    try:
        while True:
            attempt_count += 1
            active_traj_id = env.traj_id
            try:
                res, success_judge, used_steps, rrt_error = solve(
                    env,
                    seed=seed,
                    debug=False,
                    vis=args.vis,
                    artifact_layout=artifact_layout,
                    push_test_count=args.push_test_count,
                )
            except Exception as exc:
                print(
                    "Cannot find valid solution because of an error in "
                    f"motion planning solution: {exc}"
                )
                raise

            # Discard trajectories where simulator contact invalidates an
            # otherwise successful RRT plan.
            if rrt_error:
                seed += seed_stride
                retry_count += 1
                if retry_count >= MAX_RETRY_COUNT:
                    raise RuntimeError(
                        f"Trajectory {active_traj_id} failed "
                        f"{MAX_RETRY_COUNT} consecutive attempts."
                    )

                _discard_failed_attempt(
                    env,
                    artifact_layout,
                    active_traj_id,
                )
                continue

            if res == -1:
                success = False
                failed_motion_plans += 1
            else:
                success = bool(success_judge)
                solution_episode_lengths.append(used_steps)
            successes.append(success)

            if args.save_video:
                env.flush_video(
                    dir_path=str(artifact_layout.video_dir),
                    name=active_traj_id,
                )

            env.advance_episode()
            retry_count = 0
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=float(np.mean(successes)),
                    avg_steps_length=(
                        float(np.mean(solution_episode_lengths))
                        if solution_episode_lengths
                        else 0.0
                    ),
                    failed_motion_plan_rate=(
                        failed_motion_plans / attempt_count
                    ),
                )
            )
            seed += seed_stride
            passed += 1
            if passed == args.num_traj:
                break
    finally:
        pbar.close()
        env.close()

def main(args):
    if args.run_id is None:
        args.run_id = time.strftime("%Y%m%d_%H%M%S")
    assignments = build_worker_assignments(
        num_traj=args.num_traj,
        num_procs=args.num_procs,
    )
    if len(assignments) > 1 and args.sim_backend != "cpu":
        raise ValueError(
            "--num-procs > 1 requires --sim-backend cpu"
        )
    if len(assignments) > 1 and args.vis:
        raise ValueError("--vis cannot be combined with multiple workers")

    task_layout = ArtifactLayout(
        run_id=args.run_id,
        difficulty=args.scene_difficulty,
        object_type=args.object_type,
        object_idx=args.obj_idx,
        demos_root=Path(args.record_dir),
    )
    task_layout.ensure_task_is_new()
    # Atomically claim the task before worker processes begin writing. If a
    # second invocation races with this one, exactly one mkdir can succeed.
    task_layout.demos_task_dir.mkdir(parents=True, exist_ok=False)

    proc_args = []
    for assignment in assignments:
        worker_args = deepcopy(args)
        worker_args.num_traj = assignment.num_traj
        proc_args.append(
            (
                worker_args,
                task_layout.for_worker(assignment.worker_id),
                assignment.worker_id,
                assignment.seed_start,
                assignment.seed_stride,
            )
        )

    if len(proc_args) == 1:
        _main(*proc_args[0])
    else:
        pool = mp.Pool(processes=len(proc_args))
        try:
            pool.starmap(_main, proc_args)
        except BaseException:
            pool.terminate()
            raise
        else:
            pool.close()
        finally:
            pool.join()
if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
