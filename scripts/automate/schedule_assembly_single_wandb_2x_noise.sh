#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  GPU_IDS=0,1,2,3 scripts/automate/schedule_assembly_single_wandb_2x_noise.sh
  GPU_IDS=6 TASK_IDS="00004 00007" scripts/automate/schedule_assembly_single_wandb_2x_noise.sh

Options:
  --list              Print the task queue after applying TASK_IDS, then exit.
  -h, --help          Show this help.

Environment:
  GPU_IDS             Comma/space separated GPU list. Default: ${GPU_ID:-6}
  TASK_IDS            Optional comma/space separated assembly IDs. Default: all supported IDs.
  OUTPUT_ROOT         Root for checkpoints, W&B data, temp files, and scheduler logs.
                      Default: /mnt/nas/share2/home/lq
  LOG_DIR             Scheduler log directory. Default:
                      OUTPUT_ROOT/logs/automate_assembly_2x_noise_scheduler/<timestamp>
  POLL_SECONDS        Heartbeat interval while waiting for running jobs. Default: 30
  DRY_RUN             If true, print the schedule without launching training.
  SKIP_COMPLETED      If true, omit tasks with an ep_<MAX_ITERATIONS> checkpoint.
  RUN_NAME_PREFIX     Prefix for RUN_NAME. Default:
                      OUTPUT_ROOT/logs/rl_games/Assembly/automate_assembly
  WANDB_RUN_NAME_PREFIX
                      Prefix for WANDB_RUN_NAME. Default: automate_assembly
  TRAIN_SCRIPT        Single-task training script to launch.

Per task, this scheduler sets:
  GPU_ID, ASSEMBLY_ID, DISASSEMBLY_TRAJ, RUN_NAME, WANDB_RUN_NAME
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

  if ((${#id} < 5)); then
    printf "%05d" "$((10#${id}))"
  else
    printf "%s" "${id}"
  fi
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

sanitize_for_filename() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  printf "%s" "${value}"
}

handle_signal() {
  local signal="$1"

  if ((STOPPING == 0)); then
    echo
    echo "[Scheduler] Received ${signal}; stopping new launches and terminating active jobs..."
  fi
  STOPPING=1

  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done

  if [[ -n "${CURRENT_TIMER_PID:-}" ]] && kill -0 "${CURRENT_TIMER_PID}" 2>/dev/null; then
    kill "${CURRENT_TIMER_PID}" 2>/dev/null || true
  fi
}

remove_active_pid() {
  local finished_pid="$1"
  local pid
  local remaining=()

  for pid in "${ACTIVE_PIDS[@]}"; do
    if [[ "${pid}" != "${finished_pid}" ]]; then
      remaining+=("${pid}")
    fi
  done

  ACTIVE_PIDS=("${remaining[@]}")
}

make_run_name() {
  local prefix="$1"
  local assembly_id="$2"

  printf "%s_%s_2x_noise" "${prefix}" "${assembly_id}"
}

launch_task() {
  local assembly_id="$1"
  local gpu_id="$2"
  local safe_gpu_id
  local log_path
  local start_time
  local run_name
  local wandb_run_name
  local pid

  safe_gpu_id="$(sanitize_for_filename "${gpu_id}")"
  log_path="${LOG_DIR}/${assembly_id}_gpu${safe_gpu_id}.log"
  start_time="$(date -Is)"
  run_name="$(make_run_name "${RUN_NAME_PREFIX}" "${assembly_id}")"
  wandb_run_name="$(make_run_name "${WANDB_RUN_NAME_PREFIX}" "${assembly_id}")"

  echo "[Scheduler] Launch assembly_id=${assembly_id} on GPU=${gpu_id}; log=${log_path}"
  (
    export OUTPUT_ROOT="${OUTPUT_ROOT}"
    export GPU_ID="${gpu_id}"
    export ASSEMBLY_ID="${assembly_id}"
    export DISASSEMBLY_TRAJ="AutoMate/${assembly_id}/disassemble_traj.json"
    export RUN_NAME="${run_name}"
    export WANDB_RUN_NAME="${wandb_run_name}"
    "${TRAIN_SCRIPT_PATH}" "${assembly_id}"
  ) >"${log_path}" 2>&1 &

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
  local end_time
  local status

  end_time="$(date -Is)"
  remove_active_pid "${pid}"
  FREE_GPUS+=("${gpu_id}")

  if ((exit_code == 0)); then
    status="success"
    echo "[Scheduler] Done assembly_id=${assembly_id} on GPU=${gpu_id}"
  elif ((STOPPING)); then
    status="terminated"
    FAILURES+=("${assembly_id}")
    printf "%s\n" "${assembly_id}" >>"${FAILED_FILE}"
    echo "[Scheduler] Terminated assembly_id=${assembly_id} on GPU=${gpu_id}; exit=${exit_code}"
  else
    status="failed"
    FAILURES+=("${assembly_id}")
    printf "%s\n" "${assembly_id}" >>"${FAILED_FILE}"
    echo "[Scheduler] Failed assembly_id=${assembly_id} on GPU=${gpu_id}; exit=${exit_code}; log=${log_path}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${assembly_id}" "${gpu_id}" "${pid}" "${start_time}" "${end_time}" \
    "${status}" "${exit_code}" "${log_path}" >>"${SUMMARY_FILE}"

  unset "PID_TO_GPU[${pid}]"
  unset "PID_TO_ID[${pid}]"
  unset "PID_TO_LOG[${pid}]"
  unset "PID_TO_START[${pid}]"
}

print_heartbeat() {
  local running=()
  local pid

  for pid in "${ACTIVE_PIDS[@]}"; do
    running+=("${PID_TO_ID[${pid}]}:gpu${PID_TO_GPU[${pid}]}:pid${pid}")
  done

  echo "[Scheduler] Running ${#ACTIVE_PIDS[@]} job(s): $(join_by ', ' "${running[@]}")"
}

wait_for_activity() {
  FINISHED_PID=""
  FINISHED_STATUS=0

  local timer_pid
  local wait_status
  local wait_pids=("${ACTIVE_PIDS[@]}")

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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_assembly_single_wandb_2x_noise.sh}"
if [[ "${TRAIN_SCRIPT}" = /* ]]; then
  TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT}"
else
  TRAIN_SCRIPT_PATH="${REPO_ROOT}/${TRAIN_SCRIPT}"
fi

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

[[ -f "${TRAIN_SCRIPT_PATH}" ]] || die "Missing training script: ${TRAIN_SCRIPT_PATH}"
[[ -x "${TRAIN_SCRIPT_PATH}" ]] || die "Training script is not executable: ${TRAIN_SCRIPT_PATH}"

POLL_SECONDS="${POLL_SECONDS:-30}"
if [[ ! "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || ((POLL_SECONDS < 1)); then
  die "POLL_SECONDS must be a positive integer: ${POLL_SECONDS}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/share2/home/lq}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-${OUTPUT_ROOT}/logs/rl_games/Assembly/automate_assembly}"
WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-automate_assembly}"

GPU_IDS_RAW="${GPU_IDS:-${GPU_ID:-6}}"
declare -a GPU_IDS_ARRAY=()
split_list "${GPU_IDS_RAW}" GPU_IDS_ARRAY
((${#GPU_IDS_ARRAY[@]} > 0)) || die "GPU_IDS cannot be empty"

declare -A SEEN_GPUS=()
for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
  [[ -n "${gpu_id}" ]] || die "GPU_IDS contains an empty GPU ID"
  [[ "${gpu_id}" != *"/"* ]] || die "GPU ID cannot contain '/': ${gpu_id}"
  if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
    die "Duplicate GPU ID in GPU_IDS: ${gpu_id}"
  fi
  SEEN_GPUS["${gpu_id}"]=1
done

if ! SUPPORTED_IDS_OUTPUT="$("${TRAIN_SCRIPT_PATH}" --list)"; then
  die "Failed to list supported IDs from ${TRAIN_SCRIPT_PATH}"
fi

declare -a SUPPORTED_IDS=()
while IFS= read -r supported_id; do
  [[ -n "${supported_id}" ]] || continue
  SUPPORTED_IDS+=("${supported_id}")
done <<<"${SUPPORTED_IDS_OUTPUT}"
((${#SUPPORTED_IDS[@]} > 0)) || die "Training script did not report any supported IDs"

declare -A SUPPORTED_ID_SET=()
for supported_id in "${SUPPORTED_IDS[@]}"; do
  SUPPORTED_ID_SET["${supported_id}"]=1
done

declare -a TASK_IDS_ARRAY=()
if [[ -n "${TASK_IDS:-}" ]]; then
  declare -a RAW_TASK_IDS=()
  split_list "${TASK_IDS}" RAW_TASK_IDS
  ((${#RAW_TASK_IDS[@]} > 0)) || die "TASK_IDS is set but empty"

  for raw_task_id in "${RAW_TASK_IDS[@]}"; do
    task_id="$(normalize_assembly_id "${raw_task_id}")"
    TASK_IDS_ARRAY+=("${task_id}")
  done
else
  TASK_IDS_ARRAY=("${SUPPORTED_IDS[@]}")
fi

declare -A SEEN_TASK_IDS=()
for task_id in "${TASK_IDS_ARRAY[@]}"; do
  [[ -n "${SUPPORTED_ID_SET[${task_id}]:-}" ]] || die "Unsupported task ID: ${task_id}"
  if [[ -n "${SEEN_TASK_IDS[${task_id}]:-}" ]]; then
    die "Duplicate task ID in queue: ${task_id}"
  fi
  SEEN_TASK_IDS["${task_id}"]=1

  traj_path="${REPO_ROOT}/AutoMate/${task_id}/disassemble_traj.json"
  [[ -f "${traj_path}" ]] || die "Missing disassembly trajectory: ${traj_path}"
done

declare -a SKIPPED_IDS=()
if is_truthy "${SKIP_COMPLETED:-0}"; then
  declare -a PENDING_TASK_IDS=()
  completion_epoch="${MAX_ITERATIONS:-100}"

  for task_id in "${TASK_IDS_ARRAY[@]}"; do
    run_dir="$(make_run_name "${RUN_NAME_PREFIX}" "${task_id}")"
    if compgen -G "${run_dir}/nn/last_Assembly_ep_${completion_epoch}_*.pth" >/dev/null; then
      SKIPPED_IDS+=("${task_id}")
    else
      PENDING_TASK_IDS+=("${task_id}")
    fi
  done

  TASK_IDS_ARRAY=("${PENDING_TASK_IDS[@]}")
fi

if ((LIST_ONLY)); then
  printf "%s\n" "${TASK_IDS_ARRAY[@]}"
  exit 0
fi

LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs/automate_assembly_2x_noise_scheduler/$(date +%Y%m%d_%H%M%S)}"
if [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${REPO_ROOT}/${LOG_DIR}"
fi

echo "[Scheduler] Repository: ${REPO_ROOT}"
echo "[Scheduler] Training script: ${TRAIN_SCRIPT_PATH}"
echo "[Scheduler] Output root: ${OUTPUT_ROOT}"
echo "[Scheduler] GPUs: $(join_by ', ' "${GPU_IDS_ARRAY[@]}")"
echo "[Scheduler] Tasks: ${#TASK_IDS_ARRAY[@]}"
if ((${#SKIPPED_IDS[@]} > 0)); then
  echo "[Scheduler] Skipped completed tasks: ${#SKIPPED_IDS[@]}"
fi
echo "[Scheduler] Log directory: ${LOG_DIR}"
echo "[Scheduler] Poll seconds: ${POLL_SECONDS}"

if is_truthy "${DRY_RUN:-0}"; then
  echo "[Scheduler] Dry run: no training processes will be launched."
  for index in "${!TASK_IDS_ARRAY[@]}"; do
    gpu_id="${GPU_IDS_ARRAY[$((index % ${#GPU_IDS_ARRAY[@]}))]}"
    task_id="${TASK_IDS_ARRAY[${index}]}"
    echo "[Scheduler] Would queue assembly_id=${task_id} on GPU=${gpu_id}"
  done
  exit 0
fi

mkdir -p "${LOG_DIR}"

SUMMARY_FILE="${LOG_DIR}/summary.tsv"
FAILED_FILE="${LOG_DIR}/failed.txt"
printf "assembly_id\tgpu\tpid\tstart_time\tend_time\tstatus\texit_code\tlog_path\n" >"${SUMMARY_FILE}"
: >"${FAILED_FILE}"

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

next_index=0
task_count="${#TASK_IDS_ARRAY[@]}"

while ((next_index < task_count || ${#ACTIVE_PIDS[@]} > 0)); do
  while ((STOPPING == 0 && next_index < task_count && ${#FREE_GPUS[@]} > 0)); do
    gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    task_id="${TASK_IDS_ARRAY[${next_index}]}"
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
      echo "[Scheduler] wait returned unexpectedly; continuing to monitor active jobs." >&2
    fi
  fi
done

trap - INT TERM

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
