#!/bin/bash
#SBATCH --job-name=dllm-eval-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24   
#SBATCH --mem=128G           
#SBATCH --output=logs/eval_1gpu_%j.log
#SBATCH --error=logs/eval_1gpu_%j.err

set -eo pipefail

# ========================================================================
# --- ENVIRONMENT & DIRECTORY CONFIGURATION ---
# ========================================================================
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${PROJECT_DIR}/scratch}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_DIR}/envs/llada2.0.sif}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${SCRATCH_ROOT}"

export PYTHONPATH="${PROJECT_DIR}/third_party/dFactory:${PROJECT_DIR}/third_party/dFactory/VeOmni:${PROJECT_DIR}/third_party/dInfer/python:${PYTHONPATH}"
export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/hf_cache/datasets"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1

# Trigger the custom Trajectory Logger in dInfer
export DINFER_LOG_TRAJECTORY=1
export TORCH_COMPILE_DISABLE=1  

# Process-isolated cache directory
JOB_ID="${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/triton_cache_${JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}"

# ========================================================================
# --- CHECKPOINT SELECTION ---
# ========================================================================
ACTIVE_CKPT="sft-openhermes-bs32_baseline/checkpoints/global_step_14000"
# ACTIVE_CKPT="sft-openhermes-bs32_phase_top128/checkpoints/global_step_9000"
# ACTIVE_CKPT="sft-openhermes-bs32_phase_top96/checkpoints/global_step_9000"

TARGET_DIR="${SCRATCH_ROOT}/checkpoints/${ACTIVE_CKPT}"
RUN_NAME=$(echo "${ACTIVE_CKPT}" | cut -d'/' -f1)
STEP_NAME=$(basename "${ACTIVE_CKPT}")
TRAIN_ROOT_DIR="${SCRATCH_ROOT}/checkpoints/${RUN_NAME}"
MODEL_ASSETS_DIR="${TRAIN_ROOT_DIR}/model_assets"
BASE_MODEL_MERGED="${SCRATCH_ROOT}/checkpoints/base_model_merged"

EVAL_OUT_DIR="${PROJECT_DIR}/results/eval_metrics_1gpu/${RUN_NAME}"
mkdir -p "${EVAL_OUT_DIR}"

PHASE="sft"

# Full suite of tasks from Table 1
TASKS=(
    "arc_challenge_llada_mini"
    "arc_easy_llada_mini"
    "mmlu_llada_mini"
    "hellaswag_llada_mini"
    "piqa_llada_mini"
    "gsm8k_llada_mini"
    "humaneval_llada_mini"
    "ifeval_llada_mini"
)

# Batch size = 1 for clean hardware latency/expert measurements on a single GPU
declare -A BATCH_SIZES=(
    ["arc_easy_llada_mini"]=1
    ["arc_challenge_llada_mini"]=1
    ["mmlu_llada_mini"]=1
    ["hellaswag_llada_mini"]=1
    ["piqa_llada_mini"]=1
    ["gsm8k_llada_mini"]=1
    ["humaneval_llada_mini"]=1
    ["ifeval_llada_mini"]=1
)

declare -A GEN_LENGTHS=(
    ["arc_easy_llada_mini"]=32
    ["arc_challenge_llada_mini"]=32
    ["mmlu_llada_mini"]=32
    ["hellaswag_llada_mini"]=32
    ["piqa_llada_mini"]=32
    ["gsm8k_llada_mini"]=512
    ["humaneval_llada_mini"]=512
    ["ifeval_llada_mini"]=1024
)

if [ "$PHASE" == "sft" ]; then
    echo "Running in SFT mode: Applying chat template and using eos_id=156895"
    CHAT_TEMPLATE_FLAG="--apply_chat_template"
    EOS_ID="156895"
elif [ "$PHASE" == "cpt" ]; then
    echo "Running in CPT mode: Base model evaluation, eos_id=156892"
    CHAT_TEMPLATE_FLAG=""
    EOS_ID="156892"
else
    echo "Error: PHASE must be 'cpt' or 'sft'"
    exit 1
fi

echo "========================================================================"
echo "Starting 1-GPU Eval Pipeline (Hardware + Accuracy)"
echo "Target Checkpoint: ${ACTIVE_CKPT}"
echo "========================================================================"

run_eval() {
    local model_path=$1
    local step_name=$2

    for task in "${TASKS[@]}"; do
        echo "Evaluating ${task} on ${step_name}..."

        local current_bs="${BATCH_SIZES[$task]:-1}"
        local current_gl="${GEN_LENGTHS[$task]:-128}"

        local task_out_dir="${EVAL_OUT_DIR}/${step_name}/${task}"
        mkdir -p "${task_out_dir}"

        RAND_PORT=$((20000 + RANDOM % 10000))

        # Strictly locked to tp_size=1, gpus=0 to match paper claims
        apptainer exec --nv \
            --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
            --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
            --pwd "${PROJECT_DIR}" \
            --env HF_HOME="${HF_HOME}" \
            --env PYTHONPATH="${PYTHONPATH}" \
            --env HF_ALLOW_CODE_EVAL=1 \
            --env HF_DATASETS_TRUST_REMOTE_CODE=1 \
            --env TRANSFORMERS_TRUST_REMOTE_CODE=1 \
            --env TORCH_COMPILE_DISABLE=1 \
            --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
            --env DINFER_LOG_TRAJECTORY="1" \
            "${APPTAINER_IMAGE}" \
            python third_party/dInfer/evaluations/eval_dinfer_sglang.py \
                --tasks "${task}" \
                --confirm_run_unsafe_code \
                ${CHAT_TEMPLATE_FLAG} \
                --model dInfer_eval \
                --batch_size "${current_bs}" \
                --model_args "model_path=${model_path},gen_length=${current_gl},show_speed=True,save_dir=${task_out_dir},cache=prefix,warmup_times=0,use_cudagraph=False,use_compile=False,tp_size=1,parallel=tp,cont_weight=0,use_credit=False,prefix_look=0,after_look=0,model_type=llada2,use_bd=True,use_shift=False,eos_id=${EOS_ID},vocab_size=157184,master_port=${RAND_PORT},save_samples=False,prefilling_limit=1024,mask_id=156900,gpus=0,parallel_decoding=threshold,threshold=0.97,low_threshold=0.95,block_length=32" \
                --output_path "${task_out_dir}" \
                --include_path third_party/dInfer/evaluations/tasks
    done
}

HF_SAVE_DIR_MERGED="${TARGET_DIR}/hf_ckpt_merged"
HF_SAVE_DIR_FINAL="${TARGET_DIR}/hf_ckpt_split"

if [ ! -d "${TARGET_DIR}" ]; then
    echo "ERROR: ${TARGET_DIR} does not exist. Check your ACTIVE_CKPT path."
    exit 1
fi

echo ">>> Processing ${STEP_NAME} <<<"

if [ -d "${HF_SAVE_DIR_FINAL}" ] && [ -f "${HF_SAVE_DIR_FINAL}/config.json" ]; then
    echo "Found existing HF checkpoint at ${HF_SAVE_DIR_FINAL}. Skipping extraction and conversion."
else
    echo "Extracting production distributed checkpoint to HF format..."
    apptainer exec --nv \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
        --pwd "${PROJECT_DIR}" \
        --env PYTHONPATH="${PYTHONPATH}" \
        "${APPTAINER_IMAGE}" \
        python scripts/utils/extract_production_ckpt.py \
            "${TARGET_DIR}" \
            "${HF_SAVE_DIR_MERGED}" \
            "${BASE_MODEL_MERGED}"

    echo "Splitting MoE experts for SGLang compatibility..."
    apptainer exec --nv \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
        --pwd "${PROJECT_DIR}" \
        --env PYTHONPATH="${PYTHONPATH}" \
        "${APPTAINER_IMAGE}" \
        python third_party/dFactory/scripts/moe_convertor.py \
            --input-path "${HF_SAVE_DIR_MERGED}" \
            --output-path "${HF_SAVE_DIR_FINAL}" \
            --mode split
fi

echo "Overriding baseline config.json with Phase-MoE config..."
if [ -f "${MODEL_ASSETS_DIR}/config.json" ]; then
    cp "${MODEL_ASSETS_DIR}/config.json" "${HF_SAVE_DIR_FINAL}/config.json"
    apptainer exec \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
        --pwd "${PROJECT_DIR}" \
        "${APPTAINER_IMAGE}" \
        python -c "
import json, yaml
yaml_path = '${TRAIN_ROOT_DIR}/veomni_cli.yaml'
json_path = '${HF_SAVE_DIR_FINAL}/config.json'
try:
    with open(yaml_path, 'r') as f: y = yaml.safe_load(f)
    with open(json_path, 'r') as f: c = json.load(f)
    c['phase_moe_mode'] = y.get('train', {}).get('phase_moe_mode', 'none')
    c['phase_block_size'] = y.get('train', {}).get('phase_block_size', 32)
    with open(json_path, 'w') as f: json.dump(c, f, indent=2)
    print(f'Successfully injected phase_moe_mode={c[\"phase_moe_mode\"]} into config.json')
except Exception as e:
    print(f'Config override failed: {e}')
"
fi

# Run evaluation
run_eval "${HF_SAVE_DIR_FINAL}" "${STEP_NAME}"

# Cleanup temporary triton cache
rm -rf "${TRITON_CACHE_DIR}"

echo "========================================================================"
echo "Evaluations complete. Results saved to: ${EVAL_OUT_DIR}"
echo "========================================================================"