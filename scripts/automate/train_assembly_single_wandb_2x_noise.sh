#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[AutoMate] Failed at line ${LINENO} with exit code ${status}" >&2' ERR

SUPPORTED_ASSEMBLY_IDS=(
  00004 00007 00014 00015 00016 00021 00028 00030 00032 00042
  00062 00074 00077 00078 00081 00083 00103 00110 00117 00133
  00138 00141 00143 00163 00175 00186 00187 00190 00192 00210
  00211 00213 00255 00256 00271 00293 00296 00301 00308 00318
  00319 00320 00329 00340 00345 00346 00360 00388 00410 00417
  00422 00426 00437 00444 00446 00470 00471 00480 00486 00499
  00506 00514 00537 00553 00559 00581 00597 00614 00615 00638
  00648 00649 00652 00659 00681 00686 00700 00703 00726 00731
  00741 00755 00768 00783 00831 00855 00860 00863 01026 01029
  01036 01041 01053 01079 01092 01102 01125 01129 01132 01136
)

usage() {
  cat <<'EOF'
Usage:
  scripts/automate/train_assembly_single_wandb_2x_noise.sh ASSEMBLY_ID
  ASSEMBLY_ID=00015 scripts/automate/train_assembly_single_wandb_2x_noise.sh

Examples:
  GPU_ID=6 scripts/automate/train_assembly_single_wandb_2x_noise.sh 00015
  scripts/automate/train_assembly_single_wandb_2x_noise.sh 00103

Options:
  --assembly-id ID, --task-id ID  Select one assembly task.
  --list                         Print supported assembly IDs.
  -h, --help                     Show this help.

Environment overrides:
  CONDA_ROOT, CONDA_ENV, GPU_ID, NUM_ENVS, SEED, MAX_ITERATIONS
  DISASSEMBLY_TRAJ
  RUN_NAME, WANDB_ENTITY, WANDB_PROJECT_NAME, WANDB_RUN_NAME, WANDB_MODE
EOF
}

list_supported_ids() {
  printf "%s\n" "${SUPPORTED_ASSEMBLY_IDS[@]}"
}

is_supported_id() {
  local candidate="$1"
  local id
  for id in "${SUPPORTED_ASSEMBLY_IDS[@]}"; do
    if [[ "${id}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_assembly_id() {
  local id="$1"
  if [[ ! "${id}" =~ ^[0-9]+$ ]]; then
    echo "Assembly ID must be numeric: ${id}" >&2
    exit 2
  fi
  if ((${#id} < 5)); then
    printf "%05d" "$((10#${id}))"
  else
    printf "%s" "${id}"
  fi
}

ENV_ASSEMBLY_ID="${ASSEMBLY_ID:-}"
REQUESTED_ASSEMBLY_ID=""
while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list)
      list_supported_ids
      exit 0
      ;;
    --assembly-id|--task-id)
      shift
      if (($# == 0)); then
        echo "Missing value for assembly ID option." >&2
        usage >&2
        exit 2
      fi
      REQUESTED_ASSEMBLY_ID="$1"
      ;;
    --assembly-id=*|--task-id=*)
      REQUESTED_ASSEMBLY_ID="${1#*=}"
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${REQUESTED_ASSEMBLY_ID}" ]]; then
        echo "Only one assembly ID can be selected per run." >&2
        exit 2
      fi
      REQUESTED_ASSEMBLY_ID="$1"
      ;;
  esac
  shift
done

ASSEMBLY_ID="$(normalize_assembly_id "${REQUESTED_ASSEMBLY_ID:-${ENV_ASSEMBLY_ID:-00015}}")"
if ! is_supported_id "${ASSEMBLY_ID}"; then
  echo "Unsupported assembly_id=${ASSEMBLY_ID}" >&2
  echo "Use --list to see the supported single-task IDs." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Editable defaults.
CONDA_ROOT="${CONDA_ROOT:-/mnt/nas/share/home/lq/miniconda3}"
CONDA_ENV="${CONDA_ENV:-automate}"
GPU_ID="${GPU_ID:-6}"
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
DISASSEMBLY_TRAJ="${DISASSEMBLY_TRAJ:-AutoMate/${ASSEMBLY_ID}/disassemble_traj.json}"
RUN_NAME="${RUN_NAME:-automate_assembly_${ASSEMBLY_ID}_2x_noise}"
WANDB_ENTITY="${WANDB_ENTITY:-qili0502-zhejiang-university}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-automate}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${WANDB_ENTITY}" ]]; then
  echo "Set WANDB_ENTITY to your W&B entity slug before running." >&2
  echo "It is the first path segment in a run URL: https://wandb.ai/<entity>/<project>/runs/..." >&2
  echo "Example: WANDB_ENTITY=my-team $0 ${ASSEMBLY_ID}" >&2
  exit 2
fi

cd "${REPO_ROOT}"

if [[ "${DISASSEMBLY_TRAJ}" == *"://"* || "${DISASSEMBLY_TRAJ}" = /* ]]; then
  DISASSEMBLY_TRAJ_DISPLAY="${DISASSEMBLY_TRAJ}"
else
  DISASSEMBLY_TRAJ_DISPLAY="${REPO_ROOT}/${DISASSEMBLY_TRAJ}"
fi

echo "[AutoMate] Repository: ${REPO_ROOT}"
echo "[AutoMate] Training assembly policy for assembly_id=${ASSEMBLY_ID}"
echo "[AutoMate] Disassembly trajectory: ${DISASSEMBLY_TRAJ_DISPLAY}"
echo "[AutoMate] W&B: entity=${WANDB_ENTITY}, project=${WANDB_PROJECT_NAME}, run=${WANDB_RUN_NAME}, mode=${WANDB_MODE}"
echo "[AutoMate] GPU: CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "[AutoMate] Conda env: ${CONDA_ROOT}/envs/${CONDA_ENV}"
echo "[AutoMate] Max iterations: ${MAX_ITERATIONS}"
echo "[AutoMate] 2x noise overrides enabled"

if [[ "${DISASSEMBLY_TRAJ}" != *"://"* && ! -f "${DISASSEMBLY_TRAJ}" ]]; then
  echo "Missing disassembly trajectory: ${DISASSEMBLY_TRAJ_DISPLAY}" >&2
  echo "Expected downloaded built-in demo at AutoMate/${ASSEMBLY_ID}/disassemble_traj.json, or set DISASSEMBLY_TRAJ directly." >&2
  exit 2
fi

EXPECTED_PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
CURRENT_PYTHON="$(command -v python || true)"
if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" || "${CURRENT_PYTHON}" == "${EXPECTED_PYTHON}" ]]; then
  echo "[AutoMate] Already using target conda env"
else
  echo "[AutoMate] Loading conda..."
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  echo "[AutoMate] Activating conda env..."
  conda activate "${CONDA_ENV}"
  echo "[AutoMate] Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
fi

echo "[AutoMate] Python: $(command -v python)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS="${NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS:-0}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"

echo "[AutoMate] Starting assembly training..."
echo "[AutoMate] Command: ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-AutoMate-Assembly-Direct-v0 ..."

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task=Isaac-AutoMate-Assembly-Direct-v0 \
  --num_envs="${NUM_ENVS}" \
  --seed="${SEED}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --headless \
  --track \
  --wandb-entity="${WANDB_ENTITY}" \
  --wandb-project-name="${WANDB_PROJECT_NAME}" \
  --wandb-name="${WANDB_RUN_NAME}" \
  "env.tasks.insertion.assembly_id='${ASSEMBLY_ID}'" \
  "env.tasks.insertion.disassembly_path_json='${DISASSEMBLY_TRAJ}'" \
  env.tasks.insertion.if_sbc=True \
  env.tasks.insertion.if_logging_eval=False \
  "env.tasks.insertion.fixed_asset_init_pos_noise=[0.1,0.1,0.1]" \
  env.tasks.insertion.fixed_asset_init_orn_range_deg=20.0 \
  "env.tasks.insertion.held_asset_init_pos_noise=[0.02,0.02,0.02]" \
  "env.obs_rand.fixed_asset_pos=[0.002,0.002,0.002]" \
  "agent.params.config.full_experiment_name='${RUN_NAME}'"
