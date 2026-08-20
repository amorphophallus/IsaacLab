#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  GPU_IDS=0,1,2,3 PICKLE_OUTPUT_ROOT=/path/to/automate_pickle \
    scripts/automate/schedule_generate_pickle.sh
  GPU_IDS=6 TASK_IDS="00004 00007" PICKLE_OUTPUT_ROOT=/path/to/automate_pickle \
    scripts/automate/schedule_generate_pickle.sh

Options:
  --list              Print the selected task IDs after checkpoint discovery and exclusion, then exit.
  -h, --help          Show this help.

Required environment:
  GPU_IDS             Comma/space separated physical GPU IDs. Each GPU runs at most one task.
  PICKLE_OUTPUT_ROOT  Pickle data root. It must be outside CHECKPOINT_ROOT.

Optional environment:
  TASK_IDS            Comma/space separated task IDs. Default: all discovered checkpoints.
  INCLUDE_00032       If true, include task 00032 in the selected queue.
                      Default: false
  CHECKPOINT_ROOT     Checkpoint directory. Default:
                      /mnt/nas/share2/home/lq/logs/rl_games/Assembly
  NUM_SUCCESSES       Target success pickle count per task. Default: 100
  MAX_ATTEMPTS        Attempt limit for each collector invocation. Default: 1000
  BASE_SEED           Base used to derive a task/resume-specific seed. Default: 0
  POLL_SECONDS        Heartbeat interval while waiting for jobs. Default: 30
  LOG_DIR             Scheduler log directory. Default:
                      <PICKLE_OUTPUT_ROOT>_scheduler/<timestamp>
  DRY_RUN             If true, print the resolved schedule without launching collectors.
  CONDA_ROOT          Conda installation root. Default: /mnt/nas/share/home/lq/miniconda3
  CONDA_ENV           Conda environment name. Default: automate

Advanced/testing overrides:
  ISAACLAB_LAUNCHER   Isaac Lab launcher. Default: <repo>/isaaclab.sh
  GENERATOR_SCRIPT    Pickle generator. Default: <repo>/scripts/automate/generate_pickle.py

Output layout:
  PICKLE_OUTPUT_ROOT/<assembly_id>/success/*.pkl

The task 00032 is excluded by default; set INCLUDE_00032=true to include it.
Existing success pickles are counted and only the missing amount is collected;
tasks already at or above NUM_SUCCESSES are skipped.
EOF
}

die() {
  echo "[Scheduler] $*" >&2
  exit 2
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_assembly_id() {
  local id="$1"

  if [[ ! "${id}" =~ ^[0-9]+$ ]]; then
    die "Assembly ID must be numeric: ${id}"
  fi
  if ((${#id} > 5)); then
    die "Assembly ID must contain at most five digits: ${id}"
  fi

  printf "%05d" "$((10#${id}))"
}

split_list() {
  local raw="$1"
  local -n out="$2"

  raw="${raw//,/ }"
  read -r -a out <<<"${raw}"
}

join_by() {
  local delimiter="$1"
  shift

  local first=1
  local item
  for item in "$@"; do
    if ((first)); then
      printf "%s" "${item}"
      first=0
    else
      printf "%s%s" "${delimiter}" "${item}"
    fi
  done
}

absolute_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    realpath -m -- "${path}"
  else
    realpath -m -- "${REPO_ROOT}/${path}"
  fi
}

path_is_within() {
  local child="$1"
  local parent="$2"

  case "${child}/" in
    "${parent}/"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

count_success_pickles() {
  local output_dir="$1"
  local success_dir="${output_dir}/success"
  local -a pickle_files=()

  if [[ ! -d "${success_dir}" ]]; then
    printf "0"
    return
  fi

  shopt -s nullglob
  pickle_files=("${success_dir}"/*.pkl "${success_dir}"/*.pkl.xz)
  shopt -u nullglob
  printf "%d" "${#pickle_files[@]}"
}

discover_checkpoints() {
  local checkpoint_path
  local nn_dir
  local run_dir_path
  local run_dir
  local assembly_id
  local -a checkpoint_paths=()

  shopt -s nullglob
  checkpoint_paths=(
    "${CHECKPOINT_ROOT}"/automate_assembly_*_2x_noise/nn/last_Assembly_ep_100_*.pth
  )
  shopt -u nullglob

  ((${#checkpoint_paths[@]} > 0)) || die "No ep_100 checkpoints found under ${CHECKPOINT_ROOT}"

  for checkpoint_path in "${checkpoint_paths[@]}"; do
    [[ -f "${checkpoint_path}" ]] || die "Checkpoint is not a regular file: ${checkpoint_path}"

    nn_dir="${checkpoint_path%/*}"
    run_dir_path="${nn_dir%/*}"
    run_dir="${run_dir_path##*/}"
    if [[ ! "${run_dir}" =~ ^automate_assembly_([0-9]{5})_2x_noise$ ]]; then
      die "Checkpoint parent does not match automate_assembly_<5-digit-id>_2x_noise: ${checkpoint_path}"
    fi
    assembly_id="${BASH_REMATCH[1]}"

    if [[ -n "${CHECKPOINT_BY_ID[${assembly_id}]:-}" ]]; then
      die "Multiple ep_100 checkpoints found for assembly_id=${assembly_id}: ${CHECKPOINT_BY_ID[${assembly_id}]} and ${checkpoint_path}"
    fi

    CHECKPOINT_BY_ID["${assembly_id}"]="${checkpoint_path}"
    DISCOVERED_IDS+=("${assembly_id}")
  done

  mapfile -t DISCOVERED_IDS < <(printf "%s\n" "${DISCOVERED_IDS[@]}" | sort)
}

remove_active_pid() {
  local finished_pid="$1"
  local pid
  local -a remaining=()

  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ "${pid}" != "${finished_pid}" ]]; then
      remaining+=("${pid}")
    fi
  done

  ACTIVE_PIDS=("${remaining[@]}")
}

terminate_process_group() {
  local pid="$1"

  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
}

handle_signal() {
  local signal="$1"
  local pid

  if ((STOPPING == 0)); then
    echo
    echo "[Scheduler] Received ${signal}; stopping new launches and terminating active collectors..."
  fi
  STOPPING=1

  for pid in "${ACTIVE_PIDS[@]:-}"; do
    terminate_process_group "${pid}"
  done

  if [[ -n "${CURRENT_TIMER_PID:-}" ]] && kill -0 "${CURRENT_TIMER_PID}" 2>/dev/null; then
    kill "${CURRENT_TIMER_PID}" 2>/dev/null || true
  fi
}

cleanup_active_jobs() {
  local pid

  for pid in "${ACTIVE_PIDS[@]:-}"; do
    terminate_process_group "${pid}"
  done
}

launch_task() {
  local assembly_id="$1"
  local gpu_id="$2"
  local checkpoint_path="${CHECKPOINT_BY_ID[${assembly_id}]}"
  local output_dir="${TASK_OUTPUT_DIR[${assembly_id}]}"
  local initial_successes="${INITIAL_SUCCESSES[${assembly_id}]}"
  local requested_successes="${REMAINING_SUCCESSES[${assembly_id}]}"
  local task_seed="${TASK_SEEDS[${assembly_id}]}"
  local log_path="${LOG_DIR}/${assembly_id}_gpu${gpu_id}.log"
  local start_time
  local pid
  local -a command=(
    "${ISAACLAB_LAUNCHER}"
    -p
    "${GENERATOR_SCRIPT}"
    --checkpoint "${checkpoint_path}"
    --assembly-id "${assembly_id}"
    --output-dir "${output_dir}"
    --num-successes "${requested_successes}"
    --max-attempts "${MAX_ATTEMPTS}"
    --seed "${task_seed}"
    --headless
    --device cuda:0
  )

  start_time="$(date -Is)"
  echo "[Scheduler] Launch assembly_id=${assembly_id} on GPU=${gpu_id}; existing=${initial_successes}; collect=${requested_successes}; log=${log_path}"

  {
    echo "[Collector] Assembly ID: ${assembly_id}"
    echo "[Collector] Physical GPU: ${gpu_id}; process device: cuda:0"
    echo "[Collector] Checkpoint: ${checkpoint_path}"
    echo "[Collector] Output: ${output_dir}"
    echo "[Collector] Existing successes: ${initial_successes}; requested successes: ${requested_successes}"
    echo "[Collector] Seed: ${task_seed}; max attempts: ${MAX_ATTEMPTS}"
    printf "[Collector] Command:"
    printf " %q" "${command[@]}"
    printf "\n"
  } >"${log_path}"

  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS="${NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS:-0}" \
    PYTHONUNBUFFERED=1 \
    "${command[@]}" >>"${log_path}" 2>&1 &

  pid=$!
  ACTIVE_PIDS+=("${pid}")
  PID_TO_GPU["${pid}"]="${gpu_id}"
  PID_TO_ID["${pid}"]="${assembly_id}"
  PID_TO_LOG["${pid}"]="${log_path}"
  PID_TO_START["${pid}"]="${start_time}"
}

complete_task() {
  local pid="$1"
  local exit_code="$2"
  local assembly_id="${PID_TO_ID[${pid}]}"
  local gpu_id="${PID_TO_GPU[${pid}]}"
  local log_path="${PID_TO_LOG[${pid}]}"
  local start_time="${PID_TO_START[${pid}]}"
  local checkpoint_path="${CHECKPOINT_BY_ID[${assembly_id}]}"
  local output_dir="${TASK_OUTPUT_DIR[${assembly_id}]}"
  local initial_successes="${INITIAL_SUCCESSES[${assembly_id}]}"
  local requested_successes="${REMAINING_SUCCESSES[${assembly_id}]}"
  local task_seed="${TASK_SEEDS[${assembly_id}]}"
  local final_successes
  local end_time
  local status

  final_successes="$(count_success_pickles "${output_dir}")"
  end_time="$(date -Is)"
  remove_active_pid "${pid}"
  FREE_GPUS+=("${gpu_id}")

  if ((STOPPING)); then
    status="terminated"
    FAILURES+=("${assembly_id}")
    printf "%s\n" "${assembly_id}" >>"${FAILED_FILE}"
    echo "[Scheduler] Terminated assembly_id=${assembly_id} on GPU=${gpu_id}; exit=${exit_code}; successes=${final_successes}/${NUM_SUCCESSES}"
  elif ((exit_code == 0 && final_successes >= NUM_SUCCESSES)); then
    status="success"
    echo "[Scheduler] Done assembly_id=${assembly_id} on GPU=${gpu_id}; successes=${final_successes}/${NUM_SUCCESSES}"
  else
    status="failed"
    FAILURES+=("${assembly_id}")
    printf "%s\n" "${assembly_id}" >>"${FAILED_FILE}"
    echo "[Scheduler] Failed assembly_id=${assembly_id} on GPU=${gpu_id}; exit=${exit_code}; successes=${final_successes}/${NUM_SUCCESSES}; log=${log_path}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${assembly_id}" "${gpu_id}" "${pid}" "${checkpoint_path}" "${output_dir}" \
    "${initial_successes}" "${requested_successes}" "${final_successes}" "${task_seed}" \
    "${start_time}" "${end_time}" "${status}" "${exit_code}" "${log_path}" >>"${SUMMARY_FILE}"

  unset "PID_TO_GPU[${pid}]"
  unset "PID_TO_ID[${pid}]"
  unset "PID_TO_LOG[${pid}]"
  unset "PID_TO_START[${pid}]"
}

print_heartbeat() {
  local pid
  local -a running=()

  for pid in "${ACTIVE_PIDS[@]}"; do
    running+=("${PID_TO_ID[${pid}]}:gpu${PID_TO_GPU[${pid}]}:pid${pid}")
  done

  echo "[Scheduler] Running ${#ACTIVE_PIDS[@]} collector(s): $(join_by ', ' "${running[@]}")"
}

wait_for_activity() {
  local timer_pid
  local wait_status
  local -a wait_pids=("${ACTIVE_PIDS[@]}")

  FINISHED_PID=""
  FINISHED_STATUS=0

  sleep "${POLL_SECONDS}" &
  timer_pid=$!
  CURRENT_TIMER_PID="${timer_pid}"
  wait_pids+=("${timer_pid}")

  set +e
  wait -n -p FINISHED_PID "${wait_pids[@]}"
  wait_status=$?
  set -e

  CURRENT_TIMER_PID=""

  if [[ "${FINISHED_PID:-}" == "${timer_pid}" ]]; then
    FINISHED_PID=""
    FINISHED_STATUS=0
    return 1
  fi

  if kill -0 "${timer_pid}" 2>/dev/null; then
    kill "${timer_pid}" 2>/dev/null || true
  fi
  wait "${timer_pid}" 2>/dev/null || true

  FINISHED_STATUS="${wait_status}"
  if [[ -z "${FINISHED_PID:-}" ]]; then
    return 2
  fi
  return 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIST_ONLY=0
while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list)
      LIST_ONLY=1
      ;;
    *)
      usage >&2
      die "Unknown option: $1"
      ;;
  esac
  shift
done

command -v realpath >/dev/null 2>&1 || die "The realpath command is required"

CHECKPOINT_ROOT="$(absolute_path "${CHECKPOINT_ROOT:-/mnt/nas/share2/home/lq/logs/rl_games/Assembly}")"
if [[ -z "${PICKLE_OUTPUT_ROOT:-}" ]]; then
  die "PICKLE_OUTPUT_ROOT must be explicitly set to a location outside the repository and checkpoint tree"
fi
PICKLE_OUTPUT_ROOT="$(absolute_path "${PICKLE_OUTPUT_ROOT}")"
ISAACLAB_LAUNCHER="$(absolute_path "${ISAACLAB_LAUNCHER:-${REPO_ROOT}/isaaclab.sh}")"
GENERATOR_SCRIPT="$(absolute_path "${GENERATOR_SCRIPT:-${SCRIPT_DIR}/generate_pickle.py}")"

[[ -d "${CHECKPOINT_ROOT}" ]] || die "Checkpoint root does not exist: ${CHECKPOINT_ROOT}"
[[ "${PICKLE_OUTPUT_ROOT}" != "/" ]] || die "PICKLE_OUTPUT_ROOT cannot be the filesystem root"
if path_is_within "${PICKLE_OUTPUT_ROOT}" "${CHECKPOINT_ROOT}" || path_is_within "${CHECKPOINT_ROOT}" "${PICKLE_OUTPUT_ROOT}"; then
  die "Checkpoint and pickle roots must be independent directory trees: checkpoint=${CHECKPOINT_ROOT}, output=${PICKLE_OUTPUT_ROOT}"
fi

NUM_SUCCESSES_RAW="${NUM_SUCCESSES:-100}"
MAX_ATTEMPTS_RAW="${MAX_ATTEMPTS:-1000}"
BASE_SEED_RAW="${BASE_SEED:-0}"
POLL_SECONDS_RAW="${POLL_SECONDS:-30}"

[[ "${NUM_SUCCESSES_RAW}" =~ ^[0-9]+$ ]] \
  || die "NUM_SUCCESSES must be a positive integer: ${NUM_SUCCESSES_RAW}"
[[ "${MAX_ATTEMPTS_RAW}" =~ ^[0-9]+$ ]] \
  || die "MAX_ATTEMPTS must be a positive integer: ${MAX_ATTEMPTS_RAW}"
[[ "${BASE_SEED_RAW}" =~ ^[0-9]+$ ]] \
  || die "BASE_SEED must be a non-negative integer: ${BASE_SEED_RAW}"
[[ "${POLL_SECONDS_RAW}" =~ ^[0-9]+$ ]] \
  || die "POLL_SECONDS must be a positive integer: ${POLL_SECONDS_RAW}"

NUM_SUCCESSES=$((10#${NUM_SUCCESSES_RAW}))
MAX_ATTEMPTS=$((10#${MAX_ATTEMPTS_RAW}))
BASE_SEED=$((10#${BASE_SEED_RAW}))
POLL_SECONDS=$((10#${POLL_SECONDS_RAW}))

((NUM_SUCCESSES > 0)) || die "NUM_SUCCESSES must be positive: ${NUM_SUCCESSES_RAW}"
((MAX_ATTEMPTS > 0)) || die "MAX_ATTEMPTS must be positive: ${MAX_ATTEMPTS_RAW}"
((POLL_SECONDS > 0)) || die "POLL_SECONDS must be positive: ${POLL_SECONDS_RAW}"

declare -A CHECKPOINT_BY_ID=()
declare -a DISCOVERED_IDS=()
discover_checkpoints

declare -a REQUESTED_IDS=()
if [[ -n "${TASK_IDS:-}" ]]; then
  declare -a RAW_TASK_IDS=()
  declare -A SEEN_REQUESTED_IDS=()
  split_list "${TASK_IDS}" RAW_TASK_IDS
  ((${#RAW_TASK_IDS[@]} > 0)) || die "TASK_IDS is set but empty"

  for raw_task_id in "${RAW_TASK_IDS[@]}"; do
    task_id="$(normalize_assembly_id "${raw_task_id}")"
    if [[ -n "${SEEN_REQUESTED_IDS[${task_id}]:-}" ]]; then
      die "Duplicate task ID in TASK_IDS: ${task_id}"
    fi
    SEEN_REQUESTED_IDS["${task_id}"]=1
    REQUESTED_IDS+=("${task_id}")
  done
else
  REQUESTED_IDS=("${DISCOVERED_IDS[@]}")
fi

declare -a SELECTED_IDS=()
declare -a EXCLUDED_IDS=()
for task_id in "${REQUESTED_IDS[@]}"; do
  if [[ "${task_id}" == "00032" ]] && ! is_truthy "${INCLUDE_00032:-0}"; then
    EXCLUDED_IDS+=("${task_id}")
    continue
  fi
  [[ -n "${CHECKPOINT_BY_ID[${task_id}]:-}" ]] \
    || die "No unique ep_100 checkpoint found for requested assembly_id=${task_id} under ${CHECKPOINT_ROOT}"
  SELECTED_IDS+=("${task_id}")
done

if ((LIST_ONLY)); then
  echo "[Scheduler] Discovered checkpoints: ${#DISCOVERED_IDS[@]}; excluded: ${#EXCLUDED_IDS[@]}; selected: ${#SELECTED_IDS[@]}" >&2
  if ((${#SELECTED_IDS[@]} > 0)); then
    printf "%s\n" "${SELECTED_IDS[@]}"
  fi
  exit 0
fi

if [[ -z "${GPU_IDS+x}" || -z "${GPU_IDS//[[:space:],]/}" ]]; then
  die "GPU_IDS must be explicitly set, for example GPU_IDS=0,1,2,3"
fi

declare -a GPU_IDS_ARRAY=()
declare -A SEEN_GPUS=()
split_list "${GPU_IDS}" GPU_IDS_ARRAY
((${#GPU_IDS_ARRAY[@]} > 0)) || die "GPU_IDS cannot be empty"
for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
  [[ "${gpu_id}" =~ ^(0|[1-9][0-9]*)$ ]] || die "GPU ID must be a canonical non-negative integer: ${gpu_id}"
  if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
    die "Duplicate GPU ID in GPU_IDS: ${gpu_id}"
  fi
  SEEN_GPUS["${gpu_id}"]=1
done

declare -A TASK_OUTPUT_DIR=()
declare -A INITIAL_SUCCESSES=()
declare -A REMAINING_SUCCESSES=()
declare -A TASK_SEEDS=()
declare -a PENDING_IDS=()
declare -a COMPLETED_IDS=()

for task_id in "${SELECTED_IDS[@]}"; do
  output_dir="${PICKLE_OUTPUT_ROOT}/${task_id}"
  existing_successes="$(count_success_pickles "${output_dir}")"
  task_decimal=$((10#${task_id}))
  task_seed=$((BASE_SEED + task_decimal * 1000 + existing_successes))
  ((task_seed <= 2147483647)) \
    || die "Derived seed exceeds 2147483647 for assembly_id=${task_id}: ${task_seed}"

  TASK_OUTPUT_DIR["${task_id}"]="${output_dir}"
  INITIAL_SUCCESSES["${task_id}"]="${existing_successes}"
  TASK_SEEDS["${task_id}"]="${task_seed}"

  if ((existing_successes >= NUM_SUCCESSES)); then
    REMAINING_SUCCESSES["${task_id}"]=0
    COMPLETED_IDS+=("${task_id}")
  else
    REMAINING_SUCCESSES["${task_id}"]=$((NUM_SUCCESSES - existing_successes))
    PENDING_IDS+=("${task_id}")
  fi
done

LOG_DIR="$(absolute_path "${LOG_DIR:-${PICKLE_OUTPUT_ROOT}_scheduler/$(date +%Y%m%d_%H%M%S)}")"

echo "[Scheduler] Repository: ${REPO_ROOT}"
echo "[Scheduler] Checkpoint root: ${CHECKPOINT_ROOT}"
echo "[Scheduler] Pickle output root: ${PICKLE_OUTPUT_ROOT}"
echo "[Scheduler] GPUs: $(join_by ', ' "${GPU_IDS_ARRAY[@]}")"
echo "[Scheduler] Discovered checkpoints: ${#DISCOVERED_IDS[@]}"
if ((${#EXCLUDED_IDS[@]} > 0)); then
  echo "[Scheduler] Excluded task IDs: $(join_by ', ' "${EXCLUDED_IDS[@]}")"
fi
echo "[Scheduler] Selected tasks: ${#SELECTED_IDS[@]}; already complete: ${#COMPLETED_IDS[@]}; pending: ${#PENDING_IDS[@]}"
echo "[Scheduler] Target successes per task: ${NUM_SUCCESSES}; max attempts per invocation: ${MAX_ATTEMPTS}"
echo "[Scheduler] Log directory: ${LOG_DIR}"

if is_truthy "${DRY_RUN:-0}"; then
  echo "[Scheduler] Dry run: no directories or collector processes will be created."
  for task_id in "${COMPLETED_IDS[@]}"; do
    echo "[Scheduler] Would skip assembly_id=${task_id}; existing=${INITIAL_SUCCESSES[${task_id}]}/${NUM_SUCCESSES}"
  done
  for index in "${!PENDING_IDS[@]}"; do
    task_id="${PENDING_IDS[${index}]}"
    gpu_id="${GPU_IDS_ARRAY[$((index % ${#GPU_IDS_ARRAY[@]}))]}"
    echo "[Scheduler] Would enqueue assembly_id=${task_id}; preview_gpu=${gpu_id}; existing=${INITIAL_SUCCESSES[${task_id}]}; collect=${REMAINING_SUCCESSES[${task_id}]}; seed=${TASK_SEEDS[${task_id}]}; checkpoint=${CHECKPOINT_BY_ID[${task_id}]}; output=${TASK_OUTPUT_DIR[${task_id}]}"
  done
  exit 0
fi

command -v flock >/dev/null 2>&1 || die "The flock command is required"
if ((${#PENDING_IDS[@]} > 0)); then
  command -v setsid >/dev/null 2>&1 || die "The setsid command is required"
  [[ -x "${ISAACLAB_LAUNCHER}" ]] || die "Isaac Lab launcher is not executable: ${ISAACLAB_LAUNCHER}"
  [[ -f "${GENERATOR_SCRIPT}" ]] || die "Missing pickle generator: ${GENERATOR_SCRIPT}"
fi

mkdir -p "${PICKLE_OUTPUT_ROOT}" "${LOG_DIR}"
LOCK_FILE="${PICKLE_OUTPUT_ROOT}/.schedule_generate_pickle.lock"
exec {LOCK_FD}>"${LOCK_FILE}"
if ! flock -n "${LOCK_FD}"; then
  die "Another pickle scheduler is already using ${PICKLE_OUTPUT_ROOT}"
fi

SUMMARY_FILE="${LOG_DIR}/summary.tsv"
FAILED_FILE="${LOG_DIR}/failed.txt"
printf "assembly_id\tgpu\tpid\tcheckpoint\toutput_dir\tinitial_successes\trequested_successes\tfinal_successes\tseed\tstart_time\tend_time\tstatus\texit_code\tlog_path\n" >"${SUMMARY_FILE}"
: >"${FAILED_FILE}"

for task_id in "${COMPLETED_IDS[@]}"; do
  timestamp="$(date -Is)"
  printf "%s\t-\t-\t%s\t%s\t%s\t0\t%s\t-\t%s\t%s\talready_complete\t0\t-\n" \
    "${task_id}" "${CHECKPOINT_BY_ID[${task_id}]}" "${TASK_OUTPUT_DIR[${task_id}]}" \
    "${INITIAL_SUCCESSES[${task_id}]}" "${INITIAL_SUCCESSES[${task_id}]}" \
    "${timestamp}" "${timestamp}" >>"${SUMMARY_FILE}"
done

if ((${#PENDING_IDS[@]} == 0)); then
  echo "[Scheduler] Every selected task already has at least ${NUM_SUCCESSES} success pickles."
  echo "[Scheduler] Summary: ${SUMMARY_FILE}"
  exit 0
fi

CONDA_ROOT="${CONDA_ROOT:-/mnt/nas/share/home/lq/miniconda3}"
CONDA_ENV="${CONDA_ENV:-automate}"
EXPECTED_PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
CURRENT_PYTHON="$(command -v python || true)"
if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" || "${CURRENT_PYTHON}" == "${EXPECTED_PYTHON}" ]]; then
  echo "[Scheduler] Already using conda env ${CONDA_ENV}"
else
  [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] \
    || die "Missing conda initialization script: ${CONDA_ROOT}/etc/profile.d/conda.sh"
  [[ -x "${EXPECTED_PYTHON}" ]] || die "Missing target Python: ${EXPECTED_PYTHON}"
  echo "[Scheduler] Activating conda env ${CONDA_ENV}"
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
echo "[Scheduler] Python: $(command -v python)"

cd "${REPO_ROOT}"

declare -a ACTIVE_PIDS=()
declare -a FREE_GPUS=("${GPU_IDS_ARRAY[@]}")
declare -a FAILURES=()
declare -A PID_TO_GPU=()
declare -A PID_TO_ID=()
declare -A PID_TO_LOG=()
declare -A PID_TO_START=()

STOPPING=0
CURRENT_TIMER_PID=""
FINISHED_PID=""
FINISHED_STATUS=0

trap 'handle_signal SIGINT' INT
trap 'handle_signal SIGTERM' TERM
trap cleanup_active_jobs EXIT

next_index=0
task_count="${#PENDING_IDS[@]}"

while ((next_index < task_count || ${#ACTIVE_PIDS[@]} > 0)); do
  while ((STOPPING == 0 && next_index < task_count && ${#FREE_GPUS[@]} > 0)); do
    gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    task_id="${PENDING_IDS[${next_index}]}"
    ((next_index += 1))
    launch_task "${task_id}" "${gpu_id}"
  done

  if ((${#ACTIVE_PIDS[@]} == 0)); then
    break
  fi

  if wait_for_activity; then
    complete_task "${FINISHED_PID}" "${FINISHED_STATUS}"
  else
    wait_result=$?
    if ((wait_result == 1)); then
      print_heartbeat
    elif ((STOPPING)); then
      continue
    else
      echo "[Scheduler] wait returned unexpectedly; continuing to monitor active collectors." >&2
    fi
  fi
done

trap - INT TERM EXIT

if ((STOPPING)); then
  echo "[Scheduler] Stopped before completing the full queue."
  echo "[Scheduler] Summary: ${SUMMARY_FILE}"
  echo "[Scheduler] Failed/terminated tasks: ${FAILED_FILE}"
  exit 130
fi

echo "[Scheduler] All queued tasks finished."
echo "[Scheduler] Summary: ${SUMMARY_FILE}"

if ((${#FAILURES[@]} > 0)); then
  echo "[Scheduler] Failed tasks: $(join_by ', ' "${FAILURES[@]}")"
  echo "[Scheduler] Failed task list: ${FAILED_FILE}"
  exit 1
fi

echo "[Scheduler] No failed tasks."
exit 0
