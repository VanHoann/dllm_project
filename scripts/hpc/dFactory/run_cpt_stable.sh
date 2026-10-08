#!/bin/bash
set -e

# ==============================================================================
# --- USER CONFIGURATION & ENVIRONMENT -----------------------------------------
# ==============================================================================
BLOCK_SIZE=32          
MAX_STEPS=250       
MODE="warmup"  

# Project root dynamically resolved relative to this script, or via environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Working directories (defaults to local workspace folders)
WORK_DIR="${WORK_DIR:-${PROJECT_DIR}/workspace}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PROJECT_DIR}/checkpoints/Ling-mini-2.0-to-LLaDA2.0-mini-merged}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_DIR}/envs/llada2.0.sif}"

# Checkpoint loading paths (set as environment variables or leave empty)
LOAD_CKPT="${LOAD_CKPT:-}"
RESUME_2D_FROM_1D_CKPT="false"

# --- MoE Architecture Toggle ---
# "baseline" -> Standard DeepSeek auxiliary-loss-free balancing
# "phase"    -> Phase-Localized Expert Pools (Soft Shaping / Hard Pruning)
MOE_TYPE="phase"

# --- Phase-MoE Specific Configuration (Only applies if MOE_TYPE="phase") ---
PHASE_MOE_MODE="hard"        # "none", "soft", "hard", "frozen"
PHASE_BLOCK_SIZE=32          # Fixed block size for density calculation
PHASE_LAMBDA_LB=0.001       # Update rate for Load Balancing push

# Parameters for "soft" mode
# PHASE_LAMBDA_SPEC=0.002      # Update rate for Phase Specialization pull
# PHASE_GAUSSIAN_SIGMA=1.5     # Standard deviation for 1D smoothing blur
# PHASE_KERNEL_SIZE=5          # Window size for 1D Gaussian kernel

# Parameters for "hard" mode
INIT_HARD_FROM_SOFT="true"
HARD_PRUNE_K=96
HARD_PRUNE_STATS_PATH="${HARD_PRUNE_STATS_PATH:-${PROJECT_DIR}/checkpoints/moe_stats_step_7000.pt}"
HARD_PRUNE_STRATEGY="activation" # "activation" or "affinity" 
# ==============================================================================

# Python Environment Setup
export PYTHONPATH="${PROJECT_DIR}/third_party/dFactory:${PROJECT_DIR}/third_party/dFactory/VeOmni:${PROJECT_DIR}/third_party/dInfer/python:${PYTHONPATH}"
export HF_HOME="${WORK_DIR}/hf_cache"

# Determine Python Training Script
if [ "$MOE_TYPE" = "phase" ]; then
    TRAIN_SCRIPT="third_party/dFactory/tasks/train_llada2_bd_with_dparallel_phase_moe.py"
    WANDB_PROJECT="dllm-phase3-phase-moe-${MODE}"
else
    TRAIN_SCRIPT="third_party/dFactory/tasks/train_llada2_bd_with_dparallel.py"
    WANDB_PROJECT="dllm-phase3-baseline-${MODE}"
fi

# WandB Setup (Default to offline to prevent auth failures and reviewer identity leaks)
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_NAME="cpt-${MODE}-bs${BLOCK_SIZE}"
export WANDB_DIR="${WORK_DIR}/wandb_logs"
mkdir -p "$WANDB_DIR"

# Define Output Directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "$MOE_TYPE" = "phase" ]; then
    RUN_SUFFIX="_${MOE_TYPE}_${PHASE_MOE_MODE}"
else
    RUN_SUFFIX="_${MOE_TYPE}"
fi

CKPT_ROOT="${WORK_DIR}/checkpoints/cpt-${MODE}-bs${BLOCK_SIZE}${RUN_SUFFIX}_${TIMESTAMP}"
mkdir -p "${CKPT_ROOT}/checkpoints"
mkdir -p "${PROJECT_DIR}/logs"

LOG_FILE="${PROJECT_DIR}/logs/cpt_${MODE}_bs${BLOCK_SIZE}${RUN_SUFFIX}_${TIMESTAMP}.log"

# Define dynamic config target
CONFIG_FILE="${PROJECT_DIR}/scripts/hpc/dFactory/cpt_${MODE}_bs${BLOCK_SIZE}.yaml"
mkdir -p "$(dirname "$CONFIG_FILE")"

# ==============================================================================
# --- DYNAMIC CONFIG INJECTION ---
# ==============================================================================
DYNAMIC_ASSETS_DIR="${CKPT_ROOT}/model_assets_init"
mkdir -p "${DYNAMIC_ASSETS_DIR}"

if [ -d "${BASE_MODEL_DIR}" ]; then
    cp -r "${BASE_MODEL_DIR}/"* "${DYNAMIC_ASSETS_DIR}/"
fi

if [ "$MOE_TYPE" = "phase" ] && [ -f "${DYNAMIC_ASSETS_DIR}/config.json" ]; then
    echo "Injecting Phase-MoE config into dynamic config.json..."
    python3 -c "
import json
json_path = '${DYNAMIC_ASSETS_DIR}/config.json'
try:
    with open(json_path, 'r') as f: c = json.load(f)
    c['phase_moe_mode'] = '${PHASE_MOE_MODE}'
    c['phase_block_size'] = ${PHASE_BLOCK_SIZE}
    c['phase_bins'] = ${PHASE_BLOCK_SIZE} + 1
    with open(json_path, 'w') as f: json.dump(c, f, indent=2)
    print(f'Successfully injected phase_moe_mode={c[\"phase_moe_mode\"]}!')
except Exception as e:
    print(f'Config injection failed: {e}')
"
fi

echo "========================================================================"
echo "Preparing YAML Config: cpt_${MODE}_bs${BLOCK_SIZE}.yaml"
echo "Architecture Mode: ${MOE_TYPE^^}"
echo "========================================================================"

cat <<EOF > "$CONFIG_FILE"
model:
  config_path: ${DYNAMIC_ASSETS_DIR}
  model_path: ${BASE_MODEL_DIR}
  tokenizer_path: ${BASE_MODEL_DIR}
  attn_implementation: sdpa
  moe_implementation: fused
data:
  train_path: ${DATA_DIR}/smollm_cpt_arrow_stable/train
  data_type: tokenid
  datasets_type: local
  dataloader_type: native
  max_seq_len: 2048
  text_keys: input_ids
  noise_range_low: 0.1
  noise_range_high: 0.15
  num_workers: 8
train:
  output_dir: ${CKPT_ROOT}
  data_parallel_mode: fsdp2
  init_device: meta                                
  broadcast_model_weights_from_rank0: true        
  tensor_parallel_size: 1
  ulysses_parallel_size: 1
  expert_parallel_size: 4
  global_batch_size: 64
  micro_batch_size: 1
  max_steps: ${MAX_STEPS}
  num_train_epochs: 1
  rmpad: false
  optimizer: adamw
  lr: 1.0e-5
  lr_warmup_ratio: 0.05
  noise_range_high_warmup_ratio: 1.0
  weight_decay: 0.1
  max_grad_norm: 1.0
  enable_mixed_precision: true
  enable_gradient_checkpointing: false
  enable_full_shard: true
  block_diffusion_mode: true
  block_size: ${BLOCK_SIZE}
  same_token_labels: true
  complementary_mask: false
  use_wandb: true
  wandb_project: "${WANDB_PROJECT}"
  wandb_name: "${WANDB_NAME}"
  ckpt_manager: dcp
  save_steps: 250
  reset_training_state: true
  skip_rows: 1328000
EOF

# Conditionally Inject Phase-MoE vs. Baseline Specifics
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
        if [ -n "$HARD_PRUNE_STATS_PATH" ]; then
            echo "  hard_prune_stats_path: ${HARD_PRUNE_STATS_PATH}" >> "$CONFIG_FILE"
            echo "  hard_prune_strategy: ${HARD_PRUNE_STRATEGY}" >> "$CONFIG_FILE"
        fi
    fi
else
    echo "  phase_block_size: ${PHASE_BLOCK_SIZE}" >> "$CONFIG_FILE"
fi

# Append loading path if it was specified
if [ -n "$LOAD_CKPT" ]; then
    echo "  load_checkpoint_path: ${LOAD_CKPT}" >> "$CONFIG_FILE"
fi

echo "========================================================================"
echo "Launching Block Size ${BLOCK_SIZE} on 8 GPUs (Background Process)"
echo "Target Script: ${TRAIN_SCRIPT}"
echo "Log file: ${LOG_FILE}"
echo "========================================================================"

MASTER_PORT=$(shuf -i 20000-60000 -n 1)

# Ensure required run directories exist
mkdir -p "${HF_HOME}"

# Bind directories dynamically based on host execution environment
BIND_ARGS="--bind ${PROJECT_DIR}:${PROJECT_DIR},${WORK_DIR}:${WORK_DIR}"

nohup apptainer exec --nv ${BIND_ARGS} \
    --env HF_HOME="${HF_HOME}" \
    --env PYTHONPATH="${PYTHONPATH}" \
    --env WANDB_MODE="${WANDB_MODE}" \
    --env WANDB_PROJECT="${WANDB_PROJECT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "${APPTAINER_IMAGE}" \
    torchrun --nproc_per_node=8 --master-port=${MASTER_PORT} \
    "${TRAIN_SCRIPT}" "$CONFIG_FILE" > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Task started successfully with PID: ${PID}"
echo "To monitor execution output in real-time, run:"
echo "tail -f ${LOG_FILE}"
echo "========================================================================"