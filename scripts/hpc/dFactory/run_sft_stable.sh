#!/bin/bash
set -e

# ==============================================================================
# --- DIRECTORY AUTO-DETECTION & DEFAULT PATHS ---------------------------------
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set PROJECT_ROOT to the repository root directory
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Configurable paths with fallback to local project subdirectories
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PROJECT_ROOT}/models/base_model}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/openhermes_2_5_arrow/train}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${PROJECT_ROOT}/outputs}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/envs/llada2.0.sif}"
APPTAINER_BINDS="${APPTAINER_BINDS:-/tmp}"

# ==============================================================================
# --- USER CONFIGURATION -------------------------------------------------------
# ==============================================================================
BLOCK_SIZE=32
NUM_EPOCHS=1
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1

# Checkpoint to resume from (set via environment or modify below)
LOAD_CKPT="${LOAD_CKPT:-${PROJECT_ROOT}/checkpoints/cpt_hard_stage/global_step_8750}"
RESUME_2D_FROM_1D_CKPT="false"

# --- MoE Architecture Toggle ---
MOE_TYPE="phase"             # "baseline" or "phase"

# --- Phase-MoE Specific Configuration ---
PHASE_MOE_MODE="hard"        # "none", "soft", "hard", "frozen"
PHASE_BLOCK_SIZE=32
PHASE_LAMBDA_LB=0.001

# Hard mode parameters
INIT_HARD_FROM_SOFT="false"
HARD_PRUNE_K=96

# ==============================================================================
# --- ENVIRONMENT & CACHE CONFIGURATION ----------------------------------------
# ==============================================================================
export PYTHONPATH="${PROJECT_ROOT}/third_party/dFactory:${PROJECT_ROOT}/third_party/dFactory/VeOmni:${PROJECT_ROOT}/third_party/dInfer/python:${PYTHONPATH}"

export HF_HOME="${OUTPUT_BASE_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/models"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Python Training Script & Logging Naming ---
if [ "$MOE_TYPE" = "phase" ]; then
    TRAIN_SCRIPT="third_party/dFactory/tasks/train_llada2_bd_with_dparallel_phase_moe.py"
    WANDB_PROJECT="phase-moe-sft"
    RUN_SUFFIX="_${MOE_TYPE}_${PHASE_MOE_MODE}"
else
    TRAIN_SCRIPT="third_party/dFactory/tasks/train_llada2_bd_with_dparallel.py"
    WANDB_PROJECT="baseline-sft"
    RUN_SUFFIX="_${MOE_TYPE}"
fi

# Graceful WandB fallback for anonymous reviewer reproducibility
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
elif [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    echo "[Notice] W&B API key not detected. Falling back to offline mode for evaluation."
    export WANDB_MODE="offline"
fi

export WANDB_NAME="llada2-sft-bs${BLOCK_SIZE}${RUN_SUFFIX}"
export WANDB_DIR="${OUTPUT_BASE_DIR}/wandb_logs"
mkdir -p "$WANDB_DIR"

# --- LOGGING & OUTPUT SETUP ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CKPT_ROOT="${OUTPUT_BASE_DIR}/checkpoints/sft-bs${BLOCK_SIZE}${RUN_SUFFIX}_${TIMESTAMP}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/sft_bs${BLOCK_SIZE}${RUN_SUFFIX}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
mkdir -p "${CKPT_ROOT}/checkpoints"

# ==============================================================================
# --- DYNAMIC CONFIG INJECTION -------------------------------------------------
# ==============================================================================
DYNAMIC_ASSETS_DIR="${CKPT_ROOT}/model_assets_init"
mkdir -p "${DYNAMIC_ASSETS_DIR}"

if [ -d "${BASE_MODEL_DIR}" ]; then
    cp -r "${BASE_MODEL_DIR}/"* "${DYNAMIC_ASSETS_DIR}/"
fi

if [ "$MOE_TYPE" = "phase" ]; then
    echo "Injecting Phase-MoE config into dynamic config.json..."
    python3 -c "
import json, os
json_path = '${DYNAMIC_ASSETS_DIR}/config.json'
if os.path.exists(json_path):
    with open(json_path, 'r') as f: c = json.load(f)
    c['phase_moe_mode'] = '${PHASE_MOE_MODE}'
    c['phase_block_size'] = ${PHASE_BLOCK_SIZE}
    c['phase_bins'] = ${PHASE_BLOCK_SIZE} + 1
    with open(json_path, 'w') as f: json.dump(c, f, indent=2)
    print(f'Successfully configured phase_moe_mode={c[\"phase_moe_mode\"]}')
"
fi

# --- DYNAMIC YAML GENERATION ---
CONFIG_FILE="${CKPT_ROOT}/train_config_${TIMESTAMP}.yaml"

cat <<EOF > "$CONFIG_FILE"
model:
  config_path: ${DYNAMIC_ASSETS_DIR}
  model_path: ${BASE_MODEL_DIR}
  tokenizer_path: ${BASE_MODEL_DIR}
  attn_implementation: sdpa
  moe_implementation: fused
data:
  train_path: ${DATA_DIR}
  data_type: tokenid
  datasets_type: local           
  dataloader_type: native
  max_seq_len: 2048
  text_keys: input_ids
  noise_range_low: 0.1
  noise_range_high: 0.8
  num_workers: 8
train:
  output_dir: ${CKPT_ROOT}
  load_checkpoint_path: ${LOAD_CKPT}
  reset_training_state: true
  data_parallel_mode: fsdp2
  tensor_parallel_size: 1
  ulysses_parallel_size: 1
  expert_parallel_size: 4
  global_batch_size: ${GLOBAL_BATCH_SIZE}
  micro_batch_size: ${MICRO_BATCH_SIZE}
  num_train_epochs: ${NUM_EPOCHS}
  rmpad: false
  rmpad_with_pos_ids: false
  optimizer: adamw
  beta1: 0.9
  beta2: 0.999
  lr: 1.0e-5
  lr_warmup_ratio: 0.05
  noise_range_high_warmup_ratio: 1.0
  weight_decay: 0.1
  max_grad_norm: 1.0
  enable_mixed_precision: true
  enable_gradient_checkpointing: false
  enable_full_shard: true
  enable_fsdp_offload: false
  init_device: meta
  broadcast_model_weights_from_rank0: true
  save_steps: 500
  save_hf_weights: true
  block_diffusion_mode: true
  block_size: ${BLOCK_SIZE}
  same_token_labels: true
  complementary_mask: true
  use_wandb: true                 
  wandb_project: "${WANDB_PROJECT}"
  wandb_name: "${WANDB_NAME}"
  ckpt_manager: dcp 
EOF

# Conditionally Inject Phase-MoE Specifics into YAML
if [ "$MOE_TYPE" = "phase" ]; then
    echo "  phase_moe_mode: ${PHASE_MOE_MODE}" >> "$CONFIG_FILE"
    echo "  phase_block_size: ${PHASE_BLOCK_SIZE}" >> "$CONFIG_FILE"
    echo "  resume_2d_from_1d_ckpt: ${RESUME_2D_FROM_1D_CKPT}" >> "$CONFIG_FILE"
    
    if [ "$PHASE_MOE_MODE" = "soft" ]; then
        echo "  phase_lambda_lb: ${PHASE_LAMBDA_LB}" >> "$CONFIG_FILE"
        echo "  phase_lambda_spec: ${PHASE_LAMBDA_SPEC}" >> "$CONFIG_FILE"
        echo "  phase_gaussian_sigma: ${PHASE_GAUSSIAN_SIGMA}" >> "$CONFIG_FILE"
        echo "  phase_kernel_size: ${PHASE_KERNEL_SIZE}" >> "$CONFIG_FILE"
    elif [ "$PHASE_MOE_MODE" = "hard" ]; then
        echo "  phase_lambda_lb: ${PHASE_LAMBDA_LB}" >> "$CONFIG_FILE"
        echo "  init_hard_from_soft: ${INIT_HARD_FROM_SOFT}" >> "$CONFIG_FILE"
        if [ "$INIT_HARD_FROM_SOFT" = "true" ]; then
            echo "  hard_prune_k: ${HARD_PRUNE_K}" >> "$CONFIG_FILE"
        fi
    fi
else
    echo "  phase_block_size: ${PHASE_BLOCK_SIZE}" >> "$CONFIG_FILE"
fi

MASTER_PORT=$(shuf -i 20000-60000 -n 1)

echo "========================================================================"
echo "Starting Block Diffusion Training (${MOE_TYPE})"
echo "Checkpoint Output: $CKPT_ROOT"
echo "YAML Config:       $CONFIG_FILE"
echo "Log File:          $LOG_FILE"
echo "========================================================================"

# Dispatch runner (supports running inside Apptainer if image exists, otherwise native torchrun)
if [ -f "${APPTAINER_IMAGE}" ]; then
    apptainer exec --nv --bind "${APPTAINER_BINDS}" \
        --env HF_HOME="${HF_HOME}" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        --env PYTHONPATH="${PYTHONPATH}" \
        --env WANDB_MODE="${WANDB_MODE}" \
        --env WANDB_PROJECT="${WANDB_PROJECT}" \
        --env WANDB_API_KEY="${WANDB_API_KEY}" \
        --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        "${APPTAINER_IMAGE}" \
        torchrun --nproc_per_node=8 --master-port=${MASTER_PORT} \
        "${TRAIN_SCRIPT}" "$CONFIG_FILE" 2>&1 | tee "${LOG_FILE}"
else
    torchrun --nproc_per_node=8 --master-port=${MASTER_PORT} \
        "${TRAIN_SCRIPT}" "$CONFIG_FILE" 2>&1 | tee "${LOG_FILE}"
fi