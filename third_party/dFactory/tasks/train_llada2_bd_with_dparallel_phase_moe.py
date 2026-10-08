import json
import os
import time
import math
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Tuple, Optional

import math
from multiprocessing import Value
import torch
import torch.nn.functional as F
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry
ModelRegistry.register_modeling_path("models.llada2_moe")
from dataset.data_transform import process_mdm_tokenized_example, process_mdm_sft_example
from dataset import build_local_dataset
from utils.moe_monitor import MoEMonitor

logger = helper.create_logger(__name__)


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) cannot be greater than noise_range_high ({self.noise_range_high})."
            )
        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(f"noise_range_low must be between 0.0 and 1.0, got {self.noise_range_low}.")
        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(f"noise_range_high must be between 0.0 and 1.0, got {self.noise_range_high}.")


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(default=0.9, metadata={"help": "AdamW optimizer beta1."})
    beta2: float = field(default=0.999, metadata={"help": "AdamW optimizer beta2"})
    confidence_beta: float = field(default=0.0, metadata={"help": "Weight for the confidence loss entropy."})
    block_diffusion_mode: bool = field(default=False, metadata={"help": "True: use block_diffusion, False: full_attention"})
    block_size: int = field(default=32, metadata={"help": "The block size for block diffusion block size"})
    same_token_labels: bool = field(default=False, metadata={"help": "True: no shift, False: use next-token prediction shift."})
    complementary_mask: bool = field(default=True, metadata={"help": "Whether to use complementary masking."})
    reset_training_state: bool = field(default=False, metadata={"help": "Reset global_step/dataloader/lr when loading ckpt."})
    noise_range_high_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of total training steps to warmup noise_range_high using a cosine smoothstep. 0.0 disables it."}
    )
    # Phase-MoE Specific Configuration
    phase_moe_mode: Literal["none", "soft", "hard", "frozen"] = field(
        default="soft", metadata={"help": "Router behavior: none (standard), soft (Stage 1), hard (Stage 3)."}
    )
    resume_2d_from_1d_ckpt: bool = field(
        default=False, metadata={"help": "Set True to initialize as 1D to load a baseline checkpoint, then expand to 2D."}
    )
    phase_block_size: int = field(default=32, metadata={"help": "Fixed block size used to calculate phase density."})
    phase_lambda_lb: float = field(default=0.001, metadata={"help": "Update rate for Load Balancing push."})
    phase_lambda_spec: float = field(default=0.001, metadata={"help": "Update rate for Phase Specialization pull."})
    phase_gaussian_sigma: float = field(default=1.5, metadata={"help": "Standard deviation for 1D phase smoothing blur."})
    phase_kernel_size: int = field(default=5, metadata={"help": "Window size for 1D Gaussian smoothing kernel."})
    resume_2d_from_smaller_2d_ckpt: bool = field(
        default=False, metadata={"help": "Interpolate a smaller 2D phase bias (e.g. 33) to a larger one (e.g. 65)."}
    )
    old_phase_bins: int = field(
        default=33, metadata={"help": "The phase_bins size of the checkpoint being loaded."}
    )
    init_hard_from_soft: bool = field(
        default=False, metadata={"help": "Set True to prune a soft 2D phase-bias into a hard mask upon loading."}
    )
    hard_prune_k: int = field(
        default=128, metadata={"help": "Number of experts to keep per phase when init_hard_from_soft is True."}
    )
    hard_prune_stats_path: str = field(
        default="", metadata={"help": "Path to the .pt file containing activation/bias stats from moe_monitor."}
    )
    hard_prune_strategy: Literal["affinity", "activation"] = field(
        default="affinity", metadata={"help": "Pruning strategy: 'affinity' (structural bias) or 'activation' (data-driven counts)."}
    )
    def __post_init__(self):
        super().__post_init__()
        self.phase_bins = self.phase_block_size + 1


@dataclass
class Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    train: "LLaDA2TrainingArguments" = field(default_factory=LLaDA2TrainingArguments)


def block_diffusion_mask(q_idx, kv_idx, block_size, n, segment_ids=None):
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    attn_mask = block_diagonal | offset_block_causal | block_causal

    if segment_ids is not None:
        seg_q = segment_ids[q_idx]
        seg_kv = segment_ids[kv_idx]
        same_doc = (seg_q == seg_kv) & (seg_q != -1) & (seg_kv != -1)
        is_pad = (seg_q == -1) & (q_idx == kv_idx)
        doc_mask = same_doc | is_pad
        attn_mask = attn_mask & doc_mask

    return attn_mask


def compute_confidence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.device)
    valid_mask = (labels != -100)
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    predicted_tokens = torch.argmax(logits, dim=-1)
    correct_mask = (predicted_tokens == labels) & valid_mask
    if correct_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
    confidence_loss = entropy_per_token[correct_mask].mean()
    return confidence_loss

def main():
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    dynamic_noise_high = Value('d', args.data.noise_range_low)

    if args.data.data_type == "conversation":
        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            dynamic_noise_high=dynamic_noise_high,
            mask_token_id=156900, 
            complementary_mask=args.train.complementary_mask,
            block_size=args.train.block_size,
            phase_block_size=args.train.phase_block_size,
        )
    else:
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            dynamic_noise_high=dynamic_noise_high, 
            mask_token_id=156900,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 156892,
            complementary_mask=args.train.complementary_mask,
            block_size=args.train.block_size,
            phase_block_size=args.train.phase_block_size,
        )

    # Build Training Dataloader
    if args.data.datasets_type == "iterable":
        train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
    elif args.data.datasets_type == "mapping":
        train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
    else:
        train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type in ["mapping", "local"]:
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
    )
    logger.info_rank0("Prepare model")
    
    # Force Phase-MoE config updates to align with TrainingArguments
    init_phase_mode = "none" if getattr(args.train, "resume_2d_from_1d_ckpt", False) else args.train.phase_moe_mode
    
    init_phase_bins = args.train.old_phase_bins if getattr(args.train, "resume_2d_from_smaller_2d_ckpt", False) else args.train.phase_bins
    init_phase_block_size = init_phase_bins - 1

    model_config_overrides = {
        "phase_moe_mode": init_phase_mode,
        "phase_block_size": init_phase_block_size,
        "phase_bins": init_phase_bins
    }
    
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
        config_kwargs=model_config_overrides,
    )
    for k, v in model_config_overrides.items():
        setattr(model.config, k, v)
        
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0
    )

    optimizer = build_optimizer(
        model, lr=args.train.lr, betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay, fused=True, optimizer_type=args.train.optimizer,
    )

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer, train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr, lr_min=args.train.lr_min, lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio, lr_warmup_ratio=args.train.lr_warmup_ratio, lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(project=args.train.wandb_project, name=args.train.wandb_name, config={**vars(args.model), **vars(args.data), **vars(args.train)})
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config, global_batch_size=args.train.global_batch_size, rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids, empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource, dataloader=train_dataloader, data_path=args.data.train_path,
    )

    def unwrap_module(module):
        while hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        return module

    # ========================================================================
    # 1. LOAD CHECKPOINT
    # ========================================================================
    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        if args.train.reset_training_state:
            global_step = start_epoch = start_step = 0
        else:
            global_step = state["extra_state"]["global_step"]
            start_epoch = global_step // args.train.train_steps
            start_step = global_step % args.train.train_steps
            lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
            environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])
            if start_step == 0: iter(train_dataloader)
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    # ========================================================================
    # 2. EXPAND BUFFERS TO 2D & APPLY HARD PRUNING
    # ========================================================================
    
    # Pre-load the offline statistics if we are applying data-driven hard pruning
    prune_stats = None
    if getattr(args.train, "init_hard_from_soft", False) and args.train.phase_moe_mode == "hard":
        if args.train.hard_prune_stats_path:
            logger.info_rank0(f"Loading pruning stats from {args.train.hard_prune_stats_path} using strategy: '{args.train.hard_prune_strategy}'")
            prune_stats = torch.load(args.train.hard_prune_stats_path, map_location="cpu")

    moe_gates = []
    num_expanded = 0
    gate_idx = 0  # We track the MoE layer index to match the 3D stats array
    
    for name, m in model.named_modules():
        unwrapped_m = unwrap_module(m)
        if hasattr(unwrapped_m, "expert_bias") and hasattr(unwrapped_m, "routed_scaling_factor"):
            if unwrapped_m in moe_gates:
                continue

            # --- 1D to 2D BUFFER EXPANSION ---
            if getattr(args.train, "resume_2d_from_1d_ckpt", False) and args.train.phase_moe_mode != "none":
                unwrapped_m.phase_moe_mode = args.train.phase_moe_mode
                unwrapped_m.phase_block_size = args.train.phase_block_size
                unwrapped_m.phase_bins = args.train.phase_bins
                
                current_bias = unwrapped_m.expert_bias
                expected_shape = (args.train.phase_bins, model_config.num_experts)
                
                if current_bias.dim() == 1:
                    scale_factor = 8.0
                    expanded_bias = (current_bias.unsqueeze(0).expand(expected_shape) / scale_factor).clone()
                    expanded_bias = expanded_bias - expanded_bias.mean()
                    
                    del unwrapped_m.expert_bias
                    unwrapped_m.register_buffer("expert_bias", expanded_bias)
                    num_expanded += 1

            # --- 2D to LARGER 2D BUFFER INTERPOLATION ---
            if getattr(args.train, "resume_2d_from_smaller_2d_ckpt", False) and args.train.phase_moe_mode != "none":
                unwrapped_m.phase_moe_mode = args.train.phase_moe_mode
                unwrapped_m.phase_block_size = args.train.phase_block_size
                unwrapped_m.phase_bins = args.train.phase_bins
                
                current_bias = unwrapped_m.expert_bias # Shape: [old_phase_bins, num_experts]
                
                # 1. Isolate Phase 0 (Clean tokens - do not stretch this)
                bias_0 = current_bias[0:1, :]
                
                # 2. Extract and format Phase > 0 for 1D interpolation
                # Shape goes from [old_phases, experts] -> [1, experts, old_phases]
                bias_denoise = current_bias[1:, :].t().unsqueeze(0) 
                
                # 3. Interpolate strictly across the temporal dimension
                target_denoise_size = args.train.phase_block_size
                interpolated_denoise = torch.nn.functional.interpolate(
                    bias_denoise, size=target_denoise_size, mode='linear', align_corners=True
                ).squeeze(0).t() # Back to [target_denoise_size, num_experts]
                
                # 4. Re-center the interpolated block to mathematically guarantee zero-mean router inputs
                mu_row = interpolated_denoise.mean(dim=1, keepdim=True)
                mu_col = interpolated_denoise.mean(dim=0, keepdim=True)
                mu_global = interpolated_denoise.mean()
                interpolated_denoise = interpolated_denoise - mu_row - mu_col + mu_global
                
                # 5. Recombine and replace buffer
                new_bias = torch.cat([bias_0, interpolated_denoise], dim=0)
                
                del unwrapped_m.expert_bias
                unwrapped_m.register_buffer("expert_bias", new_bias)
                num_expanded += 1

            # --- HARD PRUNING FROM SOFT CKPT ---
            if getattr(args.train, "init_hard_from_soft", False) and args.train.phase_moe_mode == "hard":
                current_bias = unwrapped_m.expert_bias
                assert current_bias.dim() == 2
                
                # Determine which scores to use for finding Top-K
                scores = current_bias  # Fallback to current structural bias
                if prune_stats is not None:
                    if args.train.hard_prune_strategy == "activation":
                        # Activations are [Layers, Experts, Phase_Bins]. Transpose to [Phase_Bins, Experts]
                        scores = prune_stats["activation_counts"][gate_idx].transpose(0, 1).to(current_bias.device).float()
                    elif args.train.hard_prune_strategy == "affinity":
                        # Expert biases are [Layers, Phase_Bins, Experts]
                        scores = prune_stats["expert_biases"][gate_idx].to(current_bias.device).float()

                # 1. Find the top-K experts per phase using the selected strategy
                topk_vals, topk_indices = torch.topk(scores, k=args.train.hard_prune_k, dim=1)
                
                # 2. Create a Boolean mask of the same shape (2D: [phase_bins, num_experts])
                mask = torch.zeros_like(current_bias, dtype=torch.bool)
                mask.scatter_(1, topk_indices, True)
                
                # 3. Exempt Phase 0 (Clean Tokens) from pruning (Prefill stage, compute-bound)
                mask[0, :] = True  

                # 4. Set unselected experts to -2.0
                pruned_bias = current_bias.clone()
                pruned_bias[~mask] = -2.0
                
                # 5. Replace the buffer safely
                del unwrapped_m.expert_bias
                unwrapped_m.register_buffer("expert_bias", pruned_bias)

            moe_gates.append(unwrapped_m)
            gate_idx += 1  # Increment layer counter

    if getattr(args.train, "resume_2d_from_1d_ckpt", False) and num_expanded > 0:
        logger.info_rank0(f"Successfully expanded {num_expanded} expert_bias buffers from 1D to 2D.")
    if getattr(args.train, "init_hard_from_soft", False) and args.train.phase_moe_mode == "hard":
        logger.info_rank0(f"Successfully applied online hard pruning (Top-{args.train.hard_prune_k}) to Phase-MoE gates.")
    # ========================================================================

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}")

    if len(moe_gates) > 0:
        logger.info_rank0(f"Found {len(moe_gates)} Phase-MoE gates ready for Soft Shaping.")
        # Precompute 1D Gaussian Kernel for Phase Smoothing
        sigma = args.train.phase_gaussian_sigma
        kernel_size = args.train.phase_kernel_size
        x_kernel = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32, device=get_device_type())
        smooth_kernel = torch.exp(-x_kernel**2 / (2 * sigma**2))
        smooth_kernel = smooth_kernel / smooth_kernel.sum()
        smooth_kernel = smooth_kernel.view(1, 1, kernel_size).to(dtype=torch.float32) # Standardize dtype
    else:
        logger.warning_rank0("WARNING: No MoE gates found! Check FSDP wrapping configurations.")
        
    moe_monitor = MoEMonitor(
        model=model, num_experts=model_config.num_experts, num_layers=model_config.num_hidden_layers,
        first_k_dense_replace=model_config.first_k_dense_replace, phase_bins=args.train.phase_bins, log_dir=os.path.join(args.train.output_dir, "moe_logs")
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(args.train.train_steps, desc=f"Epoch {epoch + 1}", total=args.train.train_steps, initial=start_step, disable=args.train.local_rank != 0)
        data_iterator = iter(train_dataloader)
        
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            noise_warmup_steps = int(args.train.train_steps * args.train.noise_range_high_warmup_ratio)
            if noise_warmup_steps > 0:
                if global_step <= noise_warmup_steps:
                    progress = global_step / noise_warmup_steps
                    s_curve = 0.5 * (1.0 - math.cos(math.pi * progress))
                    new_high = args.data.noise_range_low + s_curve * (args.data.noise_range_high - args.data.noise_range_low)
                    dynamic_noise_high.value = new_high
                else:
                    dynamic_noise_high.value = args.data.noise_range_high
            else:
                dynamic_noise_high.value = args.data.noise_range_high

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                break

            total_loss = 0
            synchronize()
            start_time = time.time()
            num_accumulation_steps = len(micro_batches)
            total_consistency_loss = 0
            total_confidence_loss = 0

            # 2D Expert Tracker: [Num_Layers, Phase_Bins, Num_Experts]
            expert_counts = None
            avg_max_vio = 0.0
            avg_bias_std = 0.0
            avg_spec_intensity = 0.0
            
            abs_preclamp_min_0 = float('inf')
            abs_preclamp_max_0 = float('-inf')
            total_sat_rate_0 = 0.0
            
            abs_preclamp_min_1_32 = float('inf')
            abs_preclamp_max_1_32 = float('-inf')
            total_sat_rate_1_32 = 0.0
            avg_sat_rate_0 = 0.0
            avg_sat_rate_1_32 = 0.0
            total_support_sparsity = 0.0

            total_target_block_min = 0.0
            total_target_block_max = 0.0
            total_target_block_avg = 0.0

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                micro_batch = {k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()}

                if "mask_density" in micro_batch:
                    current_mask_density = micro_batch.pop("mask_density")
                else:
                    B_size = micro_batch["noisy_input_ids"].shape[0]
                    current_mask_density = torch.zeros(B_size, device=get_device_type())

                # Setup attention masks (same as before)
                if args.train.block_diffusion_mode:
                    noisy_input_ids = micro_batch["noisy_input_ids"]
                    clean_input_ids = micro_batch["input_ids"]
                    segment_ids = micro_batch.pop("segment_ids", None) 
                    batch_size = noisy_input_ids.shape[0]
                    seq_len = noisy_input_ids.shape[1] 
                    
                    full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
                    noisy_position_ids = torch.arange(seq_len, device=get_device_type(), dtype=torch.long)
                    clean_position_ids = torch.arange(seq_len, device=get_device_type(), dtype=torch.long)
                    position_ids = torch.cat([noisy_position_ids, clean_position_ids], dim=0).unsqueeze(0).expand(batch_size, -1).clone()
                    full_segment_ids = torch.cat([segment_ids, segment_ids], dim=1) if segment_ids is not None else None
                    
                    bd_attn_full_len = seq_len * 2
                    q_idx = torch.arange(bd_attn_full_len, device=get_device_type())[:, None]
                    kv_idx = torch.arange(bd_attn_full_len, device=get_device_type())[None, :]
                    attn_mask = torch.zeros((batch_size, 1, bd_attn_full_len, bd_attn_full_len), dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16, device=get_device_type())

                    for b in range(batch_size):
                        b_seg = full_segment_ids[b] if full_segment_ids is not None else None
                        mask_flag = block_diffusion_mask(q_idx, kv_idx, args.train.block_size, seq_len, b_seg)
                        attn_mask[b, 0].masked_fill_(mask_flag.logical_not(), float("-inf"))
                        
                    micro_batch["input_ids"] = full_input_ids
                    micro_batch["position_ids"] = position_ids
                    micro_batch["attention_mask"] = attn_mask
                else:
                    noisy_input_ids = micro_batch["noisy_input_ids"]
                    clean_input_ids = micro_batch["input_ids"]
                    segment_ids = micro_batch.pop("segment_ids", None)
                    batch_size = noisy_input_ids.shape[0]
                    seq_len = noisy_input_ids.shape[1]
                    
                    micro_batch["input_ids"] = noisy_input_ids
                    micro_batch["position_ids"] = torch.arange(seq_len, device=get_device_type(), dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                    
                    if segment_ids is not None:
                        q_idx = torch.arange(seq_len, device=get_device_type())[:, None]
                        kv_idx = torch.arange(seq_len, device=get_device_type())[None, :]
                        attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16, device=get_device_type())
                        for b in range(batch_size):
                            b_seg = segment_ids[b]
                            seg_q = b_seg[q_idx]
                            seg_kv = b_seg[kv_idx]
                            same_doc = (seg_q == seg_kv) & (seg_q != -1) & (seg_kv != -1)
                            is_pad = (seg_q == -1) & (q_idx == kv_idx)
                            attn_mask[b, 0].masked_fill_((same_doc | is_pad).logical_not(), float("-inf"))
                        micro_batch["attention_mask"] = attn_mask
                    else:
                        micro_batch["attention_mask"] = None

                labels = micro_batch.pop("labels", None)
                micro_batch.pop("mask_density", None)
                micro_batch.pop("noisy_input_ids", None) 

                # Retrieve phase mapping generated by data loader
                phase_indices = micro_batch.get("phase_indices", None)
                if phase_indices is not None:
                    if args.train.block_diffusion_mode:
                        # Clean tokens always fall into phase 0
                        clean_phase = torch.zeros_like(clean_input_ids)
                        phase_indices = torch.cat([phase_indices, clean_phase], dim=1)
                    micro_batch["phase_indices"] = phase_indices
                else:
                    phase_indices = torch.zeros_like(micro_batch["input_ids"])
                    micro_batch["phase_indices"] = phase_indices

                with model_fwd_context:
                    outputs = model(**micro_batch, use_cache=False, output_router_logits=True)
                    logits = outputs.logits
                    router_logits_tuple = outputs.router_logits # Tuple of (router_logits, topk_idx)
                    
                    with torch.no_grad():
                        if expert_counts is None:
                            expert_counts = [
                                torch.zeros((args.train.phase_bins, model_config.num_experts), dtype=torch.long, device=get_device_type()) 
                                for _ in range(len(router_logits_tuple))
                            ]
                        
                        if segment_ids is not None:
                            target_seg_ids = full_segment_ids if args.train.block_diffusion_mode else segment_ids
                            valid_mask = (target_seg_ids != -1)
                        else:
                            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 156892
                            valid_mask = (micro_batch["input_ids"] != pad_id)
                        
                        # Gather 2D Phase Counts
                        mb_unique_experts = []
                        for layer_idx, (_, topk_idx) in enumerate(router_logits_tuple):
                            v_topk = topk_idx[valid_mask].flatten() # [N_valid * top_k]
                            if v_topk.numel() > 0:
                                num_unique = torch.unique(v_topk).numel()
                                mb_unique_experts.append(num_unique)
                            v_phase = phase_indices[valid_mask].unsqueeze(1).expand(-1, topk_idx.size(-1)).flatten()
                            
                            # Flatten 2D coords into 1D for fast bincounting
                            flat_idx = v_phase * model_config.num_experts + v_topk
                            counts_1d = torch.bincount(flat_idx, minlength=args.train.phase_bins * model_config.num_experts)
                            expert_counts[layer_idx] += counts_1d.view(args.train.phase_bins, model_config.num_experts)

                        if mb_unique_experts:
                            mb_avg_sparsity = sum(mb_unique_experts) / len(mb_unique_experts)
                            total_support_sparsity += mb_avg_sparsity / num_accumulation_steps

                        # --- NEW: Per-Block Support Sparsity Tracking ---
                        mb_target_block_min = 0.0
                        mb_target_block_max = 0.0
                        mb_target_block_avg = 0.0
                        found_target_block = False
                        
                        bsz = phase_indices.shape[0]
                        seq_len = phase_indices.shape[1]
                        block_size = args.train.block_size
                        num_blocks = seq_len // block_size

                        # Iterate over sequences to find just 1 active block to inspect
                        for b in range(bsz):
                            if found_target_block:
                                break
                            
                            for blk_idx in range(num_blocks):
                                start_idx = blk_idx * block_size
                                end_idx = start_idx + block_size
                                
                                b_phases = phase_indices[b, start_idx:end_idx]
                                b_valid = valid_mask[b, start_idx:end_idx]
                                
                                # Check if this specific 32-token chunk has a phase > 0
                                if (b_phases > 0).any() and b_valid.any():
                                    block_layer_experts = []
                                    
                                    # Count unique experts across all layers for this ONE block
                                    for _, topk_idx in router_logits_tuple:
                                        b_topk = topk_idx[b, start_idx:end_idx, :] # [32, top_k]
                                        valid_topk = b_topk[b_valid].flatten()
                                        
                                        if valid_topk.numel() > 0:
                                            block_layer_experts.append(torch.unique(valid_topk).numel())
                                    
                                    if block_layer_experts:
                                        mb_target_block_min = min(block_layer_experts)
                                        mb_target_block_max = max(block_layer_experts)
                                        mb_target_block_avg = sum(block_layer_experts) / len(block_layer_experts)
                                        found_target_block = True
                                        break 
                        
                        total_target_block_min += mb_target_block_min / num_accumulation_steps
                        total_target_block_max += mb_target_block_max / num_accumulation_steps
                        total_target_block_avg += mb_target_block_avg / num_accumulation_steps

                    # Compute losses...
                    if args.train.block_diffusion_mode:
                        noisy_logits = logits[:, :noisy_input_ids.shape[1]].contiguous()
                    else:
                        noisy_logits = logits

                    confidence_loss = torch.tensor(0.0, device=noisy_logits.device)
                    if args.train.confidence_beta > 0:
                        confidence_loss = compute_confidence_loss(logits=noisy_logits, labels=labels)

                    if args.train.same_token_labels:
                        unscaled_loss = torch.nn.functional.cross_entropy(noisy_logits.view(-1, noisy_logits.shape[-1]), labels.view(-1), reduction="none").view(noisy_logits.shape[0], -1) 
                        valid_tokens = (labels != -100).sum() 
                        consistency_loss = (unscaled_loss.sum() / valid_tokens) if valid_tokens > 0 else (unscaled_loss.sum() * 0.0)
                    else:
                        shifted_noisy_logits = noisy_logits[:, :-1, :].contiguous()
                        shifted_labels = labels[:, 1:].contiguous()
                        unscaled_loss = torch.nn.functional.cross_entropy(shifted_noisy_logits.view(-1, shifted_noisy_logits.shape[-1]), shifted_labels.view(-1), reduction="none").view(shifted_noisy_logits.shape[0], -1)
                        valid_tokens = (shifted_labels != -100).sum()
                        consistency_loss = (unscaled_loss.sum() / valid_tokens) if valid_tokens > 0 else (unscaled_loss.sum() * 0.0)
                        
                combined_loss = consistency_loss + confidence_loss * args.train.confidence_beta
                loss = combined_loss / num_accumulation_steps
                with model_bwd_context:
                    loss.backward()
                moe_monitor.update_activations(
                    phase_indices=phase_indices, 
                    valid_mask=valid_mask, 
                    router_logits_tuple=router_logits_tuple
                )
                total_loss += loss.item()
                total_consistency_loss += consistency_loss.item() / num_accumulation_steps
                total_confidence_loss += confidence_loss.item() / num_accumulation_steps
                del micro_batch

            moe_monitor.update_gradients()

            # --- PHASE-MOE SHAPING (BIAS UPDATE) ---
            if expert_counts is not None and len(moe_gates) > 0:
                if args.train.phase_moe_mode in ["soft", "hard"]:
                    with torch.no_grad():
                        expert_counts_tensor = torch.stack(expert_counts).to(get_device_type())
                        
                        # 1. Synchronize token counts across EP/DP worlds
                        dist.all_reduce(expert_counts_tensor, op=dist.ReduceOp.SUM, group=None)
                        
                        total_max_vio = 0.0
                        total_bias_std = 0.0
                        total_spec_intensity = 0.0
                        
                        for i, gate in enumerate(moe_gates):
                            counts = expert_counts_tensor[i].float() 
                            M_current = gate.expert_bias.clone().float()
                            clamp_threshold = 0.4
                            
                            # ==============================================================
                            # A. HARD MODE: Active-Only Load Balancing
                            # ==============================================================
                            if args.train.phase_moe_mode == "hard":
                                active_mask = (M_current > -1.0)
                                
                                # 1. Push (Load Balancing) - Only among active experts
                                active_counts = counts * active_mask
                                num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)
                                mu_phi = active_counts.sum(dim=1, keepdim=True) / num_active
                                
                                lb_penalty = torch.sign(active_counts - mu_phi) * active_mask
                                delta_M = - (args.train.phase_lambda_lb * lb_penalty)
                                
                                M = M_current + delta_M
                                
                                # 2. Masked Double Mean-Centering
                                # Phase 0
                                act_0 = active_mask[0]
                                if act_0.any():
                                    M[0][act_0] = M[0][act_0] - M[0][act_0].mean()

                                # Denoise Phases
                                M_denoise = M[1:]
                                act_denoise = active_mask[1:]
                                
                                row_mean = (M_denoise * act_denoise).sum(dim=1, keepdim=True) / act_denoise.sum(dim=1, keepdim=True).clamp(min=1)
                                
                                M[1:] = (M_denoise - row_mean) * act_denoise
                                
                                # 3. Safety Clamp Active & Enforce Hard Mask
                                M.clamp_(min=-clamp_threshold, max=clamp_threshold)
                                M[~active_mask] = -2.0
                                
                                gate.expert_bias.copy_(M.to(gate.expert_bias.dtype))
                                
                                # For tracking stats, strictly look at the active subset
                                M_0_track = M[0][act_0] if act_0.any() else torch.tensor([], device=M.device)
                                M_denoise_track = M[1:][act_denoise] if act_denoise.any() else torch.tensor([], device=M.device)

                            # ==============================================================
                            # B. SOFT MODE: Global Shaping & Blur (Leaky Integrator)
                            # ==============================================================
                            elif args.train.phase_moe_mode == "soft":
                                # 1. --- Calculate Delta (Push and Pull) ---
                                mu_phi = counts.mean(dim=1, keepdim=True) 
                                lb_penalty = torch.sign(counts - mu_phi)
                                delta_M = - (args.train.phase_lambda_lb * lb_penalty)

                                counts_denoise = counts[1:]
                                mu_i_denoise = counts_denoise.mean(dim=0, keepdim=True)
                                spec_reward_denoise = torch.sign(counts_denoise - mu_i_denoise)

                                delta_M[1:] += (args.train.phase_lambda_spec * spec_reward_denoise)

                                # 2. --- Apply Active Mask (The Vacuum Fix) ---
                                # Silence phantom penalties in phases that received zero tokens
                                active_phase_mask = (counts.sum(dim=1, keepdim=True) > 0).float()
                                delta_M[1:] = delta_M[1:] * active_phase_mask[1:]

                                # 3. --- Add Delta to Absolute State ---
                                M_temp = M_current + delta_M
                                M = torch.empty_like(M_temp)
                                M[0] = M_temp[0] # Phase 0 (clean tokens) is isolated from the temporal blur

                                # 4. --- Spatial Blur on Absolute State ---
                                if args.train.phase_gaussian_sigma > 0:
                                    M_to_smooth = M_temp[1:].t().unsqueeze(1) # Shape: [E, 1, 32]
                                    # Use 'replicate' padding so boundaries don't bleed into 0.0
                                    M_padded = F.pad(M_to_smooth, (kernel_size//2, kernel_size//2), mode='replicate')
                                    M_smoothed = F.conv1d(M_padded, smooth_kernel.to(device=M_temp.device, dtype=M_temp.dtype))
                                    M[1:] = M_smoothed.squeeze(1).t()
                                else:
                                    M[1:] = M_temp[1:]
                                    
                                # 5. --- Double Mean-Centering ---
                                # Centering happens AFTER the blur to mathematically guarantee zero-mean inputs to the router
                                M[0] = M[0] - M[0].mean()
                                
                                M_masked = M[1:]
                                mu_row = M_masked.mean(dim=1, keepdim=True) 
                                mu_col = M_masked.mean(dim=0, keepdim=True) 
                                mu_global = M_masked.mean()                 
                                M[1:] = M_masked - mu_row - mu_col + mu_global
            
                                # 6. --- Apply the Safety Clamp ---
                                M.clamp_(min=-clamp_threshold, max=clamp_threshold)
                                
                                gate.expert_bias.copy_(M.to(gate.expert_bias.dtype))
                                
                                # Tracking variables
                                M_0_track = M[0]
                                M_denoise_track = M[1:]

                            # ==============================================================
                            # METRIC LOGGING (Calculated identically off tracked subsets)
                            # ==============================================================
                            if M_0_track.numel() > 0:
                                abs_preclamp_min_0 = min(abs_preclamp_min_0, M_0_track.min().item()) if abs_preclamp_min_0 != float('inf') else M_0_track.min().item()
                                abs_preclamp_max_0 = max(abs_preclamp_max_0, M_0_track.max().item()) if abs_preclamp_max_0 != float('-inf') else M_0_track.max().item()
                            
                            if M_denoise_track.numel() > 0:
                                abs_preclamp_min_1_32 = min(abs_preclamp_min_1_32, M_denoise_track.min().item()) if abs_preclamp_min_1_32 != float('inf') else M_denoise_track.min().item()
                                abs_preclamp_max_1_32 = max(abs_preclamp_max_1_32, M_denoise_track.max().item()) if abs_preclamp_max_1_32 != float('-inf') else M_denoise_track.max().item()
                            
                            sat_rate_0 = ((M_0_track >= clamp_threshold) | (M_0_track <= -clamp_threshold)).float().mean().item() if M_0_track.numel() > 0 else 0.0
                            sat_rate_denoise = ((M_denoise_track >= clamp_threshold) | (M_denoise_track <= -clamp_threshold)).float().mean().item() if M_denoise_track.numel() > 0 else 0.0
                            
                            total_sat_rate_0 += sat_rate_0
                            total_sat_rate_1_32 += sat_rate_denoise
                            
                            # --- Tracking Metrics (Token-Weighted Max Vio) ---
                            error = counts - mu_phi
                            active_phases = (mu_phi.squeeze() > 0)

                            phase_vios = torch.zeros(args.train.phase_bins, device=counts.device)
                            if active_phases.any():
                                phase_vios[active_phases] = (torch.abs(error[active_phases]) / mu_phi[active_phases]).max(dim=1)[0]

                            tokens_per_phase = counts.sum(dim=1)
                            total_tokens = tokens_per_phase.sum()

                            if total_tokens > 0:
                                weighted_vio = (phase_vios * tokens_per_phase).sum() / total_tokens
                                total_max_vio += weighted_vio.item()
                            
                            active_M = M[M > -1.0]
                            total_bias_std += active_M.std().item() if active_M.numel() > 1 else 0.0
                            if args.train.phase_moe_mode == "soft":
                                total_spec_intensity += M[1:].std(dim=0).mean().item()
                        
                        avg_max_vio = total_max_vio / len(moe_gates)
                        avg_bias_std = total_bias_std / len(moe_gates)
                        avg_spec_intensity = total_spec_intensity / len(moe_gates)
                        avg_sat_rate_0 = total_sat_rate_0 / len(moe_gates)
                        avg_sat_rate_1_32 = total_sat_rate_1_32 / len(moe_gates)

                elif args.train.phase_moe_mode == "frozen":
                    # DO NOT update M. 
                    with torch.no_grad():
                        expert_counts_tensor = torch.stack(expert_counts).to(get_device_type())
                        dist.all_reduce(expert_counts_tensor, op=dist.ReduceOp.SUM, group=None)
                        
                        total_max_vio = 0.0
                        total_bias_std = 0.0
                        total_spec_intensity = 0.0
                        
                        for i, gate in enumerate(moe_gates):
                            counts = expert_counts_tensor[i].float()
                            
                            # For metric tracking, only consider the active subset if pre-pruned
                            M = gate.expert_bias
                            active_mask = (M > -1.0)
                            
                            active_counts = counts * active_mask
                            num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)
                            mu_phi = active_counts.sum(dim=1, keepdim=True) / num_active

                            error = active_counts - mu_phi
                            active_phases = (mu_phi.squeeze() > 0)
                            
                            phase_vios = torch.zeros(args.train.phase_bins, device=counts.device)
                            if active_phases.any():
                                phase_vios[active_phases] = (torch.abs(error[active_phases]) / mu_phi[active_phases]).max(dim=1)[0]
                            
                            tokens_per_phase = counts.sum(dim=1)
                            total_tokens = tokens_per_phase.sum()
                            if total_tokens > 0:
                                weighted_vio = (phase_vios * tokens_per_phase).sum() / total_tokens
                                total_max_vio += weighted_vio.item()
                                
                            active_M = M[active_mask]
                            total_bias_std += active_M.std().item() if active_M.numel() > 1 else 0.0
                            # We can skip spec intensity for frozen/hard if we want, or just estimate 0
                            total_spec_intensity += 0.0 
                            
                            M_0_track = M[0][active_mask[0]] if active_mask[0].any() else torch.tensor([], device=M.device)
                            M_denoise_track = M[1:][active_mask[1:]] if active_mask[1:].any() else torch.tensor([], device=M.device)

                            if M_0_track.numel() > 0:
                                abs_preclamp_min_0 = min(abs_preclamp_min_0, M_0_track.min().item()) if abs_preclamp_min_0 != float('inf') else M_0_track.min().item()
                                abs_preclamp_max_0 = max(abs_preclamp_max_0, M_0_track.max().item()) if abs_preclamp_max_0 != float('-inf') else M_0_track.max().item()
                            if M_denoise_track.numel() > 0:
                                abs_preclamp_min_1_32 = min(abs_preclamp_min_1_32, M_denoise_track.min().item()) if abs_preclamp_min_1_32 != float('inf') else M_denoise_track.min().item()
                                abs_preclamp_max_1_32 = max(abs_preclamp_max_1_32, M_denoise_track.max().item()) if abs_preclamp_max_1_32 != float('-inf') else M_denoise_track.max().item()

                        avg_max_vio = total_max_vio / len(moe_gates)
                        avg_bias_std = total_bias_std / len(moe_gates)
                        avg_spec_intensity = total_spec_intensity / len(moe_gates)
                        
                        avg_sat_rate_0 = 0.0
                        avg_sat_rate_1_32 = 0.0
            # ---------------------------------------------------------

            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            if global_step % 20 == 0:
                data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, vio: {avg_max_vio:.3f}, spec_int: {avg_spec_intensity:.4f}")

                if args.train.global_rank == 0:
                    # 1. Log to WandB if enabled
                    if args.train.use_wandb:
                        train_metrics.update({
                            "training/loss": total_loss, 
                            "training/cons_loss": total_consistency_loss, 
                            "training/conf_loss": total_confidence_loss,  
                            "training/grad_norm": grad_norm, 
                            "training/lr": lr,
                            "training/noise_high": dynamic_noise_high.value,
                            "moe/phase_max_vio": avg_max_vio,
                            "moe/bias_std": avg_bias_std,
                            "moe/support_sparsity_U": total_support_sparsity,  
                            "moe/support_sparsity_ratio": total_support_sparsity / model_config.num_experts,

                            "moe_block/sparsity_min": total_target_block_min,
                            "moe_block/sparsity_max": total_target_block_max,
                            "moe_block/sparsity_avg": total_target_block_avg,

                            "moe_clamp/preclamp_min_phase0": abs_preclamp_min_0,
                            "moe_clamp/preclamp_max_phase0": abs_preclamp_max_0,
                            "moe_clamp/sat_rate_phase0": avg_sat_rate_0,
                            "moe_clamp/preclamp_min_denoise": abs_preclamp_min_1_32,
                            "moe_clamp/preclamp_max_denoise": abs_preclamp_max_1_32,
                            "moe_clamp/sat_rate_denoise": avg_sat_rate_1_32,
                        })
                        if args.train.phase_moe_mode == "soft":
                                train_metrics["moe/spec_intensity"] = avg_spec_intensity
                        wandb.log(train_metrics, step=global_step)

                    # 2. Injected Sanity Check Text Monitor
                    sample_idx = 0
                    
                    # Labels are -100 for unmasked tokens. Find masked targets from the last micro-batch.
                    masked_positions = (labels[sample_idx] != -100).nonzero(as_tuple=True)[0]
                    mon_input_ids = noisy_input_ids 

                    if len(masked_positions) > 0:
                        logger.info(f"\n--- [TRAIN TEXT MONITOR] Step {global_step} ---")
                        
                        # Convert logits to probabilities ONCE for the sample
                        probs = torch.nn.functional.softmax(noisy_logits[sample_idx], dim=-1)
                        num_to_inspect = min(4, len(masked_positions))
                        
                        for i in range(num_to_inspect):
                            pos = masked_positions[i].item()
                            start_ctx = max(0, pos - 25)
                            end_ctx = min(mon_input_ids.shape[1], pos + 25)
                            
                            def safe_decode_with_masks(ids, mask_id=156900):
                                chunks = []
                                current_chunk = []
                                for tid in ids:
                                    if tid == mask_id:
                                        if current_chunk:
                                            chunks.append(tokenizer.decode(current_chunk))
                                            current_chunk = []
                                        chunks.append("[MASK]")
                                    else:
                                        current_chunk.append(tid)
                                if current_chunk:
                                    chunks.append(tokenizer.decode(current_chunk))
                                return "".join(chunks)

                            # Slicing from the noisy sequence
                            left_ids = mon_input_ids[sample_idx, start_ctx:pos].tolist()
                            right_ids = mon_input_ids[sample_idx, pos+1:end_ctx].tolist()
                            
                            left_str = safe_decode_with_masks(left_ids)
                            right_str = safe_decode_with_masks(right_ids)
                            context_str = f"{left_str} [TARGET_MASK] {right_str}"
                            
                            target_id = labels[sample_idx, pos].item()
                            target_str = tokenizer.decode([target_id]).replace('\n', '\\n')
                            
                            # Top-3 predictions
                            topk_probs, topk_ids = torch.topk(probs[pos], k=3)
                            
                            logger.info(f"Context : ...{context_str.strip()}...")
                            logger.info(f"Target  : '{target_str}' (ID: {target_id})")
                            
                            pred_log = "Predict :"
                            for k in range(3):
                                p_id = topk_ids[k].item()
                                p_prob = topk_probs[k].item() * 100
                                p_str = tokenizer.decode([p_id]).replace('\n', '\\n')
                                marker = "✅" if p_id == target_id else "❌"
                                pred_log += f" | {k+1}. '{p_str}' ({p_prob:.1f}% {marker})"
                                
                            logger.info(pred_log)
                            logger.info("-" * 60)

            if global_step % 100 == 0:
                moe_monitor.save(global_step)

            data_loader_tqdm.update()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model, "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step, "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(), "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(save_checkpoint_path=save_checkpoint_path, output_dir=args.train.output_dir, ckpt_manager=args.train.ckpt_manager)
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()