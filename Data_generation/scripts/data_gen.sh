#!/bin/bash
set -euo pipefail  # Enable strict mode.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Number of worker processes used for data generation.
NUM_PROCS=${NUM_PROCS:-2}

# Number of worker processes used for data processing.
NUM_PROCS_DATA_PROCESS=${NUM_PROCS_DATA_PROCESS:-2}

DEMO_DIR=${DEMO_DIR:-scene_data_100}
OUTPUT_DIR=${OUTPUT_DIR:-processed_data_100}
EPISODES_PER_OBJECT_GENERATED=${EPISODES_PER_OBJECT_GENERATED:-10}
EPISODES_PER_OBJECT_USED=${EPISODES_PER_OBJECT_USED:-4}

run_experiment() {
    local obj_idx=$1
    local num_traj=$2
    local object_demo_dir=$3
    echo "Starting task: obj_idx=$obj_idx | num_traj=$num_traj | num_procs=$NUM_PROCS"
    python -m data_generation.run \
        --obj-idx "$obj_idx" \
        --num-traj "$num_traj" \
        --num-procs "$NUM_PROCS" \
        --demo_dir "$object_demo_dir"
}

# Generate the configured number of trajectories for objects 1 through 100.
for obj_idx in $(seq 1 3); do
    object_demo_dir="${DEMO_DIR}/object_${obj_idx}"
    run_experiment "$obj_idx" "$EPISODES_PER_OBJECT_GENERATED" "$object_demo_dir"
done

echo "All generation tasks completed. Starting post-processing..."
python -m post_process.data_process --demo_dir "$DEMO_DIR"
python -m post_process.calculate_push_dis \
    --num-procs "$NUM_PROCS_DATA_PROCESS" \
    --demo_dir "$DEMO_DIR"
python -m post_process.make_point_cloud \
    --num-procs "$NUM_PROCS_DATA_PROCESS" \
    --demo_dir "$DEMO_DIR"
python -m post_process.process_h5 \
    --processes "$NUM_PROCS_DATA_PROCESS" \
    --demo_dir "$DEMO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --episodes-per-object "$EPISODES_PER_OBJECT_USED"

echo "[$(date)] The entire pipeline completed successfully."
