#!/bin/bash
#SBATCH --job-name=dllm-batch-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/batch_eval_%j.log
#SBATCH --error=logs/batch_eval_%j.err

set -e

# ========================================================================
# --- PORTABLE ENVIRONMENT SETUP ---
# ========================================================================
# Resolve root directory dynamically from script location if not preset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Localizable directories for reviewer deployment
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/results/continuous_batching_eval}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${PROJECT_DIR}/containers/llada2.0.sif}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache}"

# Ensure output and log directories exist prior to execution
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${CACHE_DIR}/hf_home"

export PYTHONPATH="${PROJECT_DIR}/third_party/dFactory:${PROJECT_DIR}/third_party/dFactory/VeOmni:${PROJECT_DIR}/third_party/dInfer/python:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}/hf_home"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1

WORKLOAD_PATH="${DATA_DIR}/mixed_poc_workload_2.jsonl"

# Checkpoint paths defined relative to CHECKPOINT_DIR
BASE_CKPT="${CHECKPOINT_DIR}/baseline_sft"
TOP128_CKPT="${CHECKPOINT_DIR}/phase_top128_sft"
TOP96_CKPT="${CHECKPOINT_DIR}/phase_top96_sft"

# ========================================================================
# --- UNIFIED CONFIGURATION DICTIONARY ---
# Format:
# CKPT_PATH | COMP_METHOD | M_CORE | DES_MODE | LAC_MODE | DMOE_P | DMOE_GUARD | DMOE_MODE | SCHEDULERS | C_LIST | B_LIST
# ========================================================================
declare -A CONFIGS=(
    # Baseline & Proposed Co-Design
    ["baseline"]="${BASE_CKPT}|none|0|none|none|0|0|none|fcfs phase_aware|16 32 64 128|1 4 8 16 32"
    ["phase_top96"]="${TOP96_CKPT}|none|0|none|none|0|0|none|fcfs phase_aware|16 32 64 128|1 4 8 16 32"
    ["phase_top128"]="${TOP128_CKPT}|none|0|none|none|0|0|none|fcfs phase_aware|16 32 64 128|1 4 8 16 32"

    # Competitor Baselines
    # ["des_seq_m96"]="${BASE_CKPT}|des_vote|96|per_sequence|none|0|0|none|fcfs|64|1 8 16"
    # ["des_batch_m96"]="${BASE_CKPT}|des_vote|96|whole_batch|none|0|0|none|fcfs|64|8 16"
    # ["team_lac_seq"]="${BASE_CKPT}|team|0|none|lac_seq|0|0|none|fcfs|64|1 8 16"
    # ["dmoe_seq_p06"]="${BASE_CKPT}|dmoe|0|none|none|0.6|16|per_sequence|fcfs|64|1 8 16"
)

# ========================================================================
# --- EXECUTION ENGINE ---
# ========================================================================
echo "========================================================================"
echo "Starting Continuous Batching Evaluation"
echo "Project Root: ${PROJECT_DIR}"
echo "Workload:     ${WORKLOAD_PATH}"
echo "Output:       ${OUTPUT_ROOT}"
echo "========================================================================"

for CONFIG_KEY in "${!CONFIGS[@]}"; do
    IFS='|' read -r CKPT_PATH COMP_METHOD M_CORE DES_MODE LAC_MODE DMOE_P DMOE_GUARD DMOE_MODE SCHEDULER_LIST C_LIST B_LIST <<< "${CONFIGS[$CONFIG_KEY]}"
    
    for SCHEDULER in $SCHEDULER_LIST; do
        for C in $C_LIST; do
            for B_STEP in $B_LIST; do
                
                OUT_JSON="${OUTPUT_ROOT}/${CONFIG_KEY}/hardware_stats_${SCHEDULER}_C${C}_B${B_STEP}.json"
                OUT_PRED="${OUTPUT_ROOT}/${CONFIG_KEY}/predictions_${SCHEDULER}_C${C}_B${B_STEP}.jsonl"
                
                mkdir -p "$(dirname "$OUT_JSON")"
                
                echo "------------------------------------------------------------------------"
                echo "Running -> Config: $CONFIG_KEY | Sched: $SCHEDULER | C: $C | B: $B_STEP"
                echo "------------------------------------------------------------------------"
                
                # Containerized execution mounting only the current project workspace
                apptainer exec --nv \
                    --bind "${PROJECT_DIR}":/workspace \
                    --pwd /workspace \
                    --env HF_HOME="${HF_HOME}" \
                    --env PYTHONPATH="/workspace/third_party/dFactory:/workspace/third_party/dFactory/VeOmni:/workspace/third_party/dInfer/python:${PYTHONPATH}" \
                    --env TORCH_COMPILE_DISABLE=1 \
                    "${CONTAINER_IMAGE}" \
                    python "${PROJECT_DIR}/scripts/hpc/dInfer/batch/simulate_continuous_batching_2.py" \
                        --model_path "${CKPT_PATH}" \
                        --workload_path "${WORKLOAD_PATH}" \
                        --output_predictions "${OUT_PRED}" \
                        --output_hardware_json "${OUT_JSON}" \
                        --scheduler_type "${SCHEDULER}" \
                        --concurrency "${C}" \
                        --b_step "${B_STEP}" \
                        --competitor_method "${COMP_METHOD}" \
                        --des_m_core "${M_CORE}" \
                        --des_mode "${DES_MODE}" \
                        --team_lac_mode "${LAC_MODE}" \
                        --dmoe_p "${DMOE_P}" \
                        --dmoe_guard_size "${DMOE_GUARD}" \
                        --dmoe_mode "${DMOE_MODE}" \
                        --block_length 32 \
                        --gpu_id 0
                        
            done
        done
    done
done

echo "========================================================================"
echo "Evaluation sweep complete. All results aggregated in ${OUTPUT_ROOT}"
echo "========================================================================"