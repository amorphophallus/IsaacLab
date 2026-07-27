#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "[AutoMate] Failed at line ${LINENO} with exit code ${status}" >&2' ERR

# Editable defaults.
CONDA_ROOT="${CONDA_ROOT:-/mnt/nas/share/home/lq/miniconda3}"
CONDA_ENV="${CONDA_ENV:-automate}"
GPU_ID="${GPU_ID:-6}"
ASSEMBLY_ID="${ASSEMBLY_ID:-00015}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
DISASSEMBLY_DIR="${DISASSEMBLY_DIR:-.}"
RUN_NAME="${RUN_NAME:-automate_disassembly_${ASSEMBLY_ID}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[AutoMate] Repository: ${REPO_ROOT}"
echo "[AutoMate] Generating disassembly trajectories for assembly_id=${ASSEMBLY_ID}"
echo "[AutoMate] Output: ${REPO_ROOT}/${DISASSEMBLY_DIR}/${ASSEMBLY_ID}_disassemble_traj.json"
echo "[AutoMate] GPU: CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "[AutoMate] Conda env: ${CONDA_ROOT}/envs/${CONDA_ENV}"

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

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS="${NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS:-0}"

echo "[AutoMate] Starting disassembly run..."

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task=Isaac-AutoMate-Disassembly-Direct-v0 \
  --num_envs="${NUM_ENVS}" \
  --seed="${SEED}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --headless \
  "env.tasks.extraction.assembly_id='${ASSEMBLY_ID}'" \
  "env.tasks.extraction.disassembly_dir='${DISASSEMBLY_DIR}'" \
  "agent.params.config.full_experiment_name='${RUN_NAME}'"
