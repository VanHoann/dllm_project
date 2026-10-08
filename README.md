# Phase-MoE: A Co-Design to Bound the Expert Explosion in Block Diffusion Language Models

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

This repository contains the official code for the paper **Phase-MoE: A Co-Design to Bound the Expert Explosion in Block Diffusion Language Models**, provided as supplementary material for review.

Phase-MoE resolves the high-bandwidth memory (HBM) bottleneck — termed the *Expert Explosion* — that occurs when scaling Block Diffusion Language Models (BDLMs) with Mixture-of-Experts (MoE) under continuous batching. Our algorithm-system co-design breaks the Pareto conflict between memory efficiency and representational capacity through two primary components:

1. **Phase-Constrained MoE (Training):** A soft-to-hard optimization curriculum that localizes candidate expert pools to distinct temporal denoising phases while isolating compute-bound prefill operations.
2. **Phase-Aware Scheduler (Inference):** A dynamic dispatch policy that minimizes intra-batch temporal spread ($`\Delta\phi \approx 0`$), structurally bounding batch-level HBM traffic without compromising the asynchronous throughput of continuous batching.

---

## 📂 Repository Structure

The codebase is built on top of customized, highly optimized versions of the `dFactory` (training) and `dInfer` (inference/serving) frameworks.

```text
├── envs/                               # Apptainer (Singularity) container definitions
├── scripts/
│   ├── hpc/
│   │   ├── dFactory/                   # Training dispatch scripts (CPT, SFT)
│   │   └── dInfer/                     # Inference, scheduling, and hardware eval scripts
│   │       ├── batch/                  # Continuous batching engine and simulation sweeps
│   └── utils/                          # Data preparation and weight surgery tools
└── third_party/
    ├── dFactory/                       # Custom training framework for BDLMs
    │   ├── models/llada2_moe/          # Phase-MoE model architectures
    │   └── tasks/                      # Soft-to-hard MoE curriculum training loops
    └── dInfer/                         # Custom inference framework powered by SGLang
        ├── decoding/                   # Diffusion runners and parallel strategies
        └── model/                      # Inference models and competitor implementations (TEAM, DES, dMoE)

```

---

## 🛠️ Environment Setup

To ensure strict reproducibility and isolate dependencies (Python 3.11, CUDA 12.4, PyTorch, vLLM, and SGLang), we provide a fully self-contained Apptainer (Singularity) definition file.

```bash
# Build the Apptainer image from the provided definition
apptainer build envs/llada2.sif envs/llada2_main.def

```

---

## 📊 Data Preparation

To replicate the training pipeline, prepare the continuous pre-training (CPT) and supervised fine-tuning (SFT) datasets into the required Arrow format:

```bash
# 1. Prepare CPT dataset (FineWeb-Edu, Cosmopedia-v2, Python-Edu)
python scripts/utils/prepare_smollm_arrow_stable.py

# 2. Prepare SFT dataset (OpenHermes-2.5)
sbatch scripts/utils/run_prepare_openhermes.sbatch

```

---

## 🚀 Training: Phase-Constrained MoE

Training Phase-MoE follows a two-stage curriculum. Rather than relying on post-hoc inference heuristics, we embed the memory constraints directly into the router architecture during training.

### Stage 1: Soft Shaping (Continual Pre-Training)

Executes the leaky integrator mechanism, balancing the load-balancing push and phase-specialization pull to smoothly localize expert utility without causing semantic collapse.

```bash
# Configure execution inside scripts/hpc/dFactory/run_cpt_stable.sh:
# MOE_TYPE="phase"
# PHASE_MOE_MODE="soft"

bash scripts/hpc/dFactory/run_cpt_stable.sh

```

*Core implementation logic is located in `third_party/dFactory/tasks/train_llada2_bd_with_dparallel_phase_moe.py`.*

### Stage 2: Hard Pruning (Supervised Fine-Tuning)

Locks the architectural constraints by truncating the continuous bias matrix to a strict support set prior to SFT, guaranteeing a hard upper bound on memory fetches.

```bash
# Configure execution inside scripts/hpc/dFactory/run_sft_stable.sh:
# MOE_TYPE="phase"
# PHASE_MOE_MODE="hard"

bash scripts/hpc/dFactory/run_sft_stable.sh

```

---

## ⚙️ Inference: Continuous Batching & Competitor Baselines

The Phase-Aware Scheduler dynamically clusters asynchronous requests by mask density at runtime. We provide a simulation engine to benchmark Phase-MoE against standard serving (FCFS) and state-of-the-art post-hoc routing mitigations evaluated in the paper.

### 1. Prepare the Mixed Workload

Generate a semantically diverse, continuous evaluation stream covering reasoning, math, and general knowledge tasks (e.g., GSM8K, IFEval, MMLU):

```bash
python scripts/hpc/dInfer/batch/prepare_mixed_workload_batched_infer_2.py \
    --output_path data/mixed_poc_workload_2.jsonl

```

### 2. Run the Continuous Batching Simulation Sweep

Execute the continuous batching engine across multiple concurrencies and step batch sizes:

```bash
sbatch scripts/hpc/dInfer/batch/sweep_continuous_batching_2.sbatch

```

> **Note on Competitor Baselines:** Post-hoc mitigations (TEAM, DES, dMoE) can be evaluated directly using the `--competitor_method` flag in `simulate_continuous_batching_2.py` (options: `des_vote`, `team`, `dmoe`). The implementations for these interventions are located in `third_party/dInfer/python/dinfer/model/`.

---

## ⏱️ Hardware Evaluation & Benchmarks

To extract true hardware telemetry (MoE kernel latency, unique expert activations from HBM) alongside generative accuracy (ARC, MMLU, GSM8K, etc.):

```bash
sbatch scripts/hpc/dInfer/eval_loop_1gpu.sbatch
```

These scripts automatically handle extracting the distributed checkpoint, converting it to Hugging Face format via `moe_convertor.py`, injecting the correct Phase-MoE `config.json`, and executing the `lm-eval` harness.