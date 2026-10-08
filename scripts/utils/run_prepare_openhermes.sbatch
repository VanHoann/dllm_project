#!/bin/bash
#SBATCH --job-name=prep_hermes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/prep_hermes_%j.log
#SBATCH --error=logs/prep_hermes_%j.err

set -e

# Configurable paths with sensible local defaults
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/.cache}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${PROJECT_DIR}/envs/llada2.0.sif}"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/Ling-mini-2.0-to-LLaDA2.0-mini-merged}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/openhermes_2_5_arrow}"

export HF_HOME="${CACHE_DIR}/hf_cache"
export HF_DATASETS_CACHE="${CACHE_DIR}/hf_cache/datasets"
export APPTAINER_TMPDIR="${CACHE_DIR}/apptainer_tmp"
export APPTAINER_CACHEDIR="${CACHE_DIR}/apptainer_cache"

mkdir -p "${PROJECT_DIR}/logs" "${CACHE_DIR}" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

# Define path for temporary python worker script
TEMP_PY_SCRIPT="scripts/utils/_temp_prepare_openhermes.py"
trap 'rm -f "${TEMP_PY_SCRIPT}"' EXIT

echo "========================================================================"
echo "Starting OpenHermes-2.5 SFT Data Preparation"
echo "Working Directory: ${PROJECT_DIR}"
echo "Model Path:        ${MODEL_PATH}"
echo "Output Directory:  ${OUTPUT_DIR}"
echo "========================================================================"

# Dynamically generate the processing script
cat << 'EOF' > "${TEMP_PY_SCRIPT}"
import os
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    model_path = os.environ.get("MODEL_PATH", "./checkpoints/Ling-mini-2.0-to-LLaDA2.0-mini-merged")
    output_dir = os.environ.get("OUTPUT_DIR", "./data/openhermes_2_5_arrow")
    max_length = 2048

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Downloading OpenHermes-2.5 dataset...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")

    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system"
    }

    def tokenize_function(example):
        messages = []
        for turn in example["conversations"]:
            role = role_map.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})

        inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        tokenized = tokenizer(
            inputs_str,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False
        )

        return {
            "input_ids": tokenized["input_ids"],
            "prompt_lengths": prompt_len,
            "seq_len": len(tokenized["input_ids"])
        }

    print("Tokenizing and formatting dataset...")
    processed_dataset = dataset.map(
        tokenize_function,
        batched=False,
        num_proc=os.cpu_count() or 16,
        remove_columns=dataset.column_names,
        desc="Pre-tokenizing OpenHermes-2.5"
    )

    total_seqs = len(processed_dataset)
    total_tokens = sum(processed_dataset["seq_len"])
    processed_dataset = processed_dataset.remove_columns(["seq_len"])

    print("\n" + "=" * 55)
    print("DATASET SUMMARY & TOKEN BUDGET")
    print("=" * 55)
    print(f"Total Sequences: {total_seqs:,}")
    print(f"Total Tokens:    {total_tokens:,} (~{total_tokens / 1e9:.2f}B tokens)")
    print("=" * 55 + "\n")

    train_output_dir = os.path.join(output_dir, "train")
    os.makedirs(train_output_dir, exist_ok=True)

    print(f"Saving Arrow dataset to: {train_output_dir}")
    processed_dataset.save_to_disk(train_output_dir)

if __name__ == "__main__":
    main()
EOF

# Execution: fallback gracefully if apptainer image is absent
if [ -f "${CONTAINER_IMAGE}" ]; then
    echo "Executing inside Apptainer container: ${CONTAINER_IMAGE}"
    apptainer exec --nv \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --pwd "${PROJECT_DIR}" \
        --env HF_HOME="${HF_HOME}" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        --env MODEL_PATH="${MODEL_PATH}" \
        --env OUTPUT_DIR="${OUTPUT_DIR}" \
        "${CONTAINER_IMAGE}" \
        python -u "${TEMP_PY_SCRIPT}"
else
    echo "Container image not found. Falling back to host Python environment..."
    python -u "${TEMP_PY_SCRIPT}"
fi