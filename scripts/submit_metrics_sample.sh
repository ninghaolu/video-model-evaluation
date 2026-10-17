#!/bin/bash
# Sample Slurm launcher for checkpoint video metrics.
#
# Copy this to a local, ignored script and replace every placeholder before
# submitting. Keep sweep-specific paths, accounts, reservations, and constraints
# out of version control.
#
#SBATCH --job-name=wm_eval_metrics
#SBATCH --output=/path/to/slurm_logs/%x_%j.out
#SBATCH --error=/path/to/slurm_logs/%x_%j.err
#SBATCH --account=REPLACE_WITH_SLURM_ACCOUNT
#SBATCH --partition=REPLACE_WITH_SLURM_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

set -eo pipefail

EVAL_ROOT="${EVAL_ROOT:-/path/to/eval/results/root}"
REPO_DIR="${REPO_DIR:-/path/to/video-model-evaluation}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/path/to/slurm_logs}"
DATASET_DIR_NAME="${DATASET_DIR_NAME:-final_dataset_eval_10_sampled}"
SUMMARY_NAME="${SUMMARY_NAME:-summary_eval.json}"
METRICS="${METRICS:-psnr,ssim,lpips,fid,fvd}"
DEVICE="${DEVICE:-auto}"
VIDEO_GLOB="${VIDEO_GLOB:-**/*.mp4}"
CONDA_ENV="${CONDA_ENV:-REPLACE_WITH_CONDA_ENV}"

RUN_JOB=0
DRY_RUN=0
INCLUDE_COMPLETED=0
LIMIT=""
EVAL_DIR=""
GENERATED_SUBDIR=""
FVD_TARGET_LEN=""

usage() {
  cat <<EOF
Usage:
  $0 [options]
  $0 --run --eval-dir /path/to/${DATASET_DIR_NAME}

Discovers ${DATASET_DIR_NAME}/videos/{gt,pred} directories under EVAL_ROOT and
submits one Slurm job per eval directory. Each job writes:
  <eval_dir>/${SUMMARY_NAME}

Options:
  --dry-run                  Print sbatch commands without submitting.
  --include-completed        Submit even if the summary already exists.
  --limit N                  Submit at most N jobs after filtering.
  --eval-root PATH           Root to search. Default: ${EVAL_ROOT}
  --eval-dir PATH            One eval dir for --run.
  --repo-dir PATH            Repository checkout. Default: ${REPO_DIR}
  --slurm-log-dir PATH       Slurm log directory. Default: ${SLURM_LOG_DIR}
  --dataset-dir-name NAME    Dataset leaf name. Default: ${DATASET_DIR_NAME}
  --summary-name NAME        Summary filename. Default: ${SUMMARY_NAME}
  --metrics LIST             Comma-separated metrics. Default: ${METRICS}
  --device DEVICE            Device passed to evaluator. Default: ${DEVICE}
  --video-glob GLOB          Video glob passed to evaluator. Default: ${VIDEO_GLOB}
  --generated-subdir NAME    Generated video dir under videos/. Default: pred.
  --fvd-target-len N         Pad/truncate FVD clips to N frames.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_JOB=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-completed)
      INCLUDE_COMPLETED=1
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --eval-root)
      EVAL_ROOT="$2"
      shift 2
      ;;
    --eval-dir)
      EVAL_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --slurm-log-dir)
      SLURM_LOG_DIR="$2"
      shift 2
      ;;
    --dataset-dir-name)
      DATASET_DIR_NAME="$2"
      shift 2
      ;;
    --summary-name)
      SUMMARY_NAME="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --video-glob)
      VIDEO_GLOB="$2"
      shift 2
      ;;
    --generated-subdir)
      GENERATED_SUBDIR="$2"
      shift 2
      ;;
    --fvd-target-len)
      FVD_TARGET_LEN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Directory does not exist: $1" >&2
    exit 2
  fi
}

has_video_pair_dirs() {
  local candidate="$1"
  [[ -d "${candidate}/videos/gt" ]] || return 1
  if [[ -n "${GENERATED_SUBDIR}" ]]; then
    [[ -d "${candidate}/videos/${GENERATED_SUBDIR}" ]]
  else
    [[ -d "${candidate}/videos/pred" ]]
  fi
}

discover_eval_dirs() {
  find "${EVAL_ROOT}" -type d -name "${DATASET_DIR_NAME}" | sort | while IFS= read -r candidate; do
    if has_video_pair_dirs "${candidate}"; then
      readlink -f "${candidate}"
    fi
  done
}

job_name_for() {
  local eval_dir="$1"
  local step_dir model_dir model_name step_name
  step_dir="$(dirname "${eval_dir}")"
  model_dir="$(dirname "${step_dir}")"
  model_name="$(basename "${model_dir}")"
  step_name="$(basename "${step_dir}")"
  printf 'wm_metrics_%s_%s\n' "${model_name}" "${step_name}" | cut -c 1-120
}

EVAL_ROOT="$(readlink -f "${EVAL_ROOT}")"
REPO_DIR="$(readlink -f "${REPO_DIR}")"
SLURM_LOG_DIR="$(readlink -m "${SLURM_LOG_DIR}")"

require_dir "${EVAL_ROOT}"
require_dir "${REPO_DIR}"

if [[ "${RUN_JOB}" -eq 0 ]]; then
  script_path="$(readlink -f "$0")"
  submitted=0
  skipped_completed=0
  skipped_limit=0
  limit_reached=0

  while IFS= read -r eval_dir; do
    if [[ "${limit_reached}" -eq 1 ]]; then
      skipped_limit=$((skipped_limit + 1))
      continue
    fi

    summary_path="${eval_dir}/${SUMMARY_NAME}"
    if [[ "${INCLUDE_COMPLETED}" -eq 0 && -f "${summary_path}" ]]; then
      echo "skip completed: ${summary_path}"
      skipped_completed=$((skipped_completed + 1))
      continue
    fi

    cmd=(
      sbatch
      --job-name="$(job_name_for "${eval_dir}")"
      --output="${SLURM_LOG_DIR}/%x_%j.out"
      --error="${SLURM_LOG_DIR}/%x_%j.err"
      "${script_path}"
      --run
      --eval-dir "${eval_dir}"
      --repo-dir "${REPO_DIR}"
      --summary-name "${SUMMARY_NAME}"
      --metrics "${METRICS}"
      --device "${DEVICE}"
      --video-glob "${VIDEO_GLOB}"
    )
    [[ -n "${GENERATED_SUBDIR}" ]] && cmd+=(--generated-subdir "${GENERATED_SUBDIR}")
    [[ -n "${FVD_TARGET_LEN}" ]] && cmd+=(--fvd-target-len "${FVD_TARGET_LEN}")

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf 'dry-run:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      mkdir -p "${SLURM_LOG_DIR}"
      "${cmd[@]}"
    fi

    submitted=$((submitted + 1))
    if [[ -n "${LIMIT}" && "${submitted}" -ge "${LIMIT}" ]]; then
      limit_reached=1
    fi
  done < <(discover_eval_dirs)

  echo "submitted=${submitted} skipped_completed=${skipped_completed} skipped_limit=${skipped_limit}"
  exit 0
fi

if [[ -z "${EVAL_DIR}" ]]; then
  echo "Need --eval-dir when running a metrics job." >&2
  exit 2
fi

EVAL_DIR="$(readlink -f "${EVAL_DIR}")"
require_dir "${EVAL_DIR}"

if ! has_video_pair_dirs "${EVAL_DIR}"; then
  echo "Expected ${EVAL_DIR}/videos/gt and ${EVAL_DIR}/videos/pred." >&2
  exit 2
fi

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

eval_cmd=(
  python -m video_model_evaluation.step_eval
  "${EVAL_DIR}"
  --summary-name "${SUMMARY_NAME}"
  --output-dir "${EVAL_DIR}/video_model_eval"
  --metrics "${METRICS}"
  --device "${DEVICE}"
  --video-glob "${VIDEO_GLOB}"
)
[[ -n "${GENERATED_SUBDIR}" ]] && eval_cmd+=(--generated-subdir "${GENERATED_SUBDIR}")
[[ -n "${FVD_TARGET_LEN}" ]] && eval_cmd+=(--fvd-target-len "${FVD_TARGET_LEN}")

"${eval_cmd[@]}"
