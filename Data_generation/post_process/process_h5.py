import argparse
import json
import multiprocessing as mp
import re
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import trimesh
from tqdm import tqdm


def _file_index(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    return int(match.group()) if match else -1


def _create_dataset(group, name: str, data) -> None:
    data = np.asarray(data)
    if data.shape == ():
        group.create_dataset(name, data=data)
    else:
        group.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=6,
        )


def _copy_npy_directory(h5f, episode_path: Path, directory_name: str) -> None:
    source_dir = episode_path / directory_name
    if not source_dir.exists():
        return

    files = sorted(source_dir.glob("*.npy"), key=_file_index)
    if not files:
        return

    group = h5f.create_group(directory_name)
    for path in files:
        _create_dataset(group, path.stem, np.load(path))


def _copy_nested_npy_directory(h5f, episode_path: Path, directory_name: str) -> None:
    source_dir = episode_path / directory_name
    if not source_dir.exists():
        return

    child_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not child_dirs:
        return

    group = h5f.create_group(directory_name)
    for child_dir in child_dirs:
        child_group = group.create_group(child_dir.name)
        for path in sorted(child_dir.glob("*.npy"), key=_file_index):
            _create_dataset(child_group, path.stem, np.load(path))


def _write_contact_points(h5f, episode_path: Path) -> None:
    contact_dir = episode_path / "contact_points"
    if not contact_dir.exists():
        return

    files = sorted(contact_dir.glob("*.ply"), key=_file_index)
    if not files:
        return

    group = h5f.create_group("contact_points")
    for path in files:
        mesh = trimesh.load(path)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        _create_dataset(group, path.stem, points)


def process_single_episode(args):
    """Export one retained trajectory to HDF5."""
    episode_path, output_dir, episode_num = args
    episode_path = Path(episode_path)

    try:
        object_name = episode_path.parent.parent.name
        episode_output_dir = output_dir / object_name / f"traj_{episode_num}"
        episode_output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = episode_output_dir / "data.h5"

        with h5py.File(h5_path, "w") as h5f:
            _write_contact_points(h5f, episode_path)

            for directory_name in (
                "object_state",
                "contact_tcp_pose",
                "push_action_pe",
                "push_action_ps",
                "push_distance",
                "world_obj_pcd",
            ):
                _copy_npy_directory(h5f, episode_path, directory_name)

            _copy_nested_npy_directory(h5f, episode_path, "local_data")

            h5f.attrs["episode_num"] = episode_num

        return {
            "state": "SUCCESS",
            "episode_num": episode_num,
            "h5_path": h5_path,
        }
    except Exception as exc:
        import traceback

        return {
            "state": "ERROR",
            "detail": f"{exc}\n{traceback.format_exc()}",
        }


def create_dataset_metadata(output_dir: Path, successful_episodes):
    episode_info = [
        {
            "episode_id": index,
            "file_path": str(result["h5_path"]),
        }
        for index, result in enumerate(successful_episodes)
    ]
    metadata = {
        "num_episodes": len(successful_episodes),
        "episode_files": [str(result["h5_path"]) for result in successful_episodes],
        "episode_info": episode_info,
    }

    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    return metadata


def parse_timestamp(name: str) -> tuple[int, int]:
    match = re.fullmatch(
        r"(\d{8})_(\d{6})(?:\.(\d+))?",
        name,
    )

    if match is None:
        return 0, 0

    date_part, time_part, process_id = match.groups()

    timestamp = int(date_part + time_part)
    process_id = int(process_id) if process_id is not None else 0

    return timestamp, process_id


def preprocess_dataset(
    demo_dir,
    output_dir,
    num_processes,
    episodes_per_object,
):
    task_dir = Path(demo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_episodes = []
    object_dirs = [path for path in task_dir.iterdir() if path.is_dir()]
    for object_dir in object_dirs:
        object_trajectories = []
        session_dirs = sorted(
            (path for path in object_dir.iterdir() if path.is_dir()),
            key=lambda path: parse_timestamp(path.name),
        )
        for session_dir in session_dirs:
            trajectories = [
                path
                for path in session_dir.glob("traj_*/")
                if path.name[5:].isdigit()
            ]
            object_trajectories.extend(
                sorted(trajectories, key=lambda path: int(path.name[5:]))
            )

        if len(object_trajectories) < episodes_per_object:
            print(
                f"Object has fewer than {episodes_per_object} trajectories: "
                f"{object_dir}"
            )
            continue

        all_episodes.extend(
            object_trajectories[:episodes_per_object]
        )

    print(f"Found {len(all_episodes)} trajectories across {len(object_dirs)} objects")
    if not all_episodes:
        print("No episodes found.")
        return

    process_args = [
        (episode_path, output_dir, episode_index)
        for episode_index, episode_path in enumerate(all_episodes)
    ]
    results = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(process_args), desc="Processing episodes") as progress:
            for result in pool.imap(process_single_episode, process_args):
                results.append(result)
                progress.update(1)

    successful = [result for result in results if result["state"] == "SUCCESS"]
    errors = [result for result in results if result["state"] == "ERROR"]
    print(f"HDF5 export complete: {len(successful)} succeeded, {len(errors)} failed")
    for error in errors[:3]:
        print(error["detail"])

    if successful:
        metadata = create_dataset_metadata(output_dir, successful)
        total_size = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
        print(f"Episodes: {metadata['num_episodes']}")
        print(f"Total size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Export retained trajectory data to HDF5")
    parser.add_argument(
        "--demo_dir",
        type=str,
        default="scene_data_100",
        help="Path to original dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data_100",
        help="Path to save HDF5 dataset",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=16,
        help="Number of worker processes",
    )
    parser.add_argument(
    "--episodes-per-object",
    type=int,
    default=500,
    help="Number of trajectories exported for each object.",
    )
    args = parser.parse_args()

    if not Path(args.demo_dir).exists():
        print(f"Input directory does not exist: {args.demo_dir}")
        return

    print(f"Using {args.processes} processes (CPU count: {mp.cpu_count()})")
    preprocess_dataset(args.demo_dir, args.output_dir, args.processes, args.episodes_per_object,)


if __name__ == "__main__":
    main()
