#!/bin/bash
set -euo pipefail  # Enable strict error handling.

export JOBLIB_MULTIPROCESSING="${JOBLIB_MULTIPROCESSING:-0}"

RUN_ID="20260928_test"
NUM_TRAJ="${NUM_TRAJ:-1}"
NUM_PROCS="${NUM_PROCS:-1}"
DIFFICULTIES=(scene_01 scene_02 scene_03 scene_04 scene_05 scene_06)


declare -Ar PUSH_TEST_COUNTS=(
    [scene_01]=1
    [scene_02]=1
    [scene_03]=1
    [scene_04]=1
    [scene_05]=30
    [scene_06]=30
)


run_experiment() {
    local obj_idx=$1 object_type=$2 scene_difficulty=$3
    local push_test_count="${PUSH_TEST_COUNTS[$scene_difficulty]:-}"
    echo "Starting task: run_id=$RUN_ID | obj_idx=$obj_idx | num_traj=$NUM_TRAJ | num_procs=$NUM_PROCS | object_type=$object_type | scene_difficulty=$scene_difficulty"
    python simulation/run.py \
        --obj-idx "$obj_idx" \
        --num-traj "$NUM_TRAJ" \
        --object_type "$object_type" \
        --scene_difficulty "$scene_difficulty" \
        --run-id "$RUN_ID" \
        --num-procs "$NUM_PROCS" \
        --push-test-count "$push_test_count" \
        --save-video
}

echo "Begin test: run_id=$RUN_ID"

for scene_difficulty in "${DIFFICULTIES[@]}"; do
    for obj_idx in $(seq 1 12); do
        run_experiment "$obj_idx" test "$scene_difficulty"
    done
    for obj_idx in $(seq 1 10); do
        run_experiment "$obj_idx" train "$scene_difficulty"
    done
done
