import json
import os
import time
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
                f"noise_range_low ({self.noise_range_low}) "
                f"cannot be greater than noise_range_high ({self.noise_range_high})."
            )

        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(
                f"noise_range_low must be between 0.0 and 1.0, but got {self.noise_range_low}."
            )

        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(
                f"noise_range_high must be between 0.0 and 1.0, but got {self.noise_range_high}."
            )


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW optimizer beta2"},
    )
    confidence_beta: float = field(
        default=0.0,
        metadata={"help": "Weight for the confidence loss entropy of correct predictions. Set to 0 to disable."},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "If train MDM in block_diffusion mode. True: use block_diffusion, False: full_attention"}
    )
    block_size: int = field(
        default=32,
        metadata={"help": "The block size for block diffusion block size"}
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."}
    )

    # --- ADDED: Complementary Mask Configuration ---
    complementary_mask: bool = field(
        default=True,
        metadata={"help": "Whether to use complementary masking for SFT data efficiency."}
    )
    reset_training_state: bool = field(
        default=False,
        metadata={"help": "Whether to reset global_step, dataloader, and lr_scheduler when loading a checkpoint."}
    )
    noise_range_high_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of total training steps to warmup noise_range_high using a cosine smoothstep. 0.0 disables it."}
    )
    phase_block_size: int = field(default=32, metadata={"help": "Fixed block size used to calculate phase density."})
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

    # --- DOCUMENT LEVEL MASKING ---
    if segment_ids is not None:
        seg_q = segment_ids[q_idx]
        seg_kv = segment_ids[kv_idx]
        
        # Must be same document, and NOT padding (-1)
        same_doc = (seg_q == seg_kv) & (seg_q != -1) & (seg_kv != -1)
        
        # Allow padding tokens (-1) to attend to themselves to prevent NaN Softmax
        is_pad = (seg_q == -1) & (q_idx == kv_idx)
        
        doc_mask = same_doc | is_pad
        attn_mask = attn_mask & doc_mask

    return attn_mask

def compute_confidence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the average entropy of the output distribution at positions where the model predicts correctly.
    """
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

    entropy_at_correct_positions = entropy_per_token[correct_mask]
    
    confidence_loss = entropy_at_correct_positions.mean()
    
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
        if not tokenizer.chat_template:
            raise ValueError(f"No chat template found in the tokenizer.")

        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            dynamic_noise_high=dynamic_noise_high, 
            mask_token_id=156900, 
            complementary_mask=args.train.complementary_mask,
            block_size=args.train.block_size
        )
    elif args.data.data_type == "tokenid":
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
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            logger.info_rank0("Start building local dataset")
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping" or args.data.datasets_type == "local":
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
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
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
        model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        # save model_assets before training
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        
        if args.train.reset_training_state:
            logger.info_rank0(f"reset_training_state is True. Model and Optimizer weights loaded from {args.train.load_checkpoint_path}, but resetting step, epoch, dataloader, and LR schedule to 0.")
            global_step = 0
            start_epoch = 0
            start_step = 0
        else:
            global_step = state["extra_state"]["global_step"]
            start_epoch = global_step // args.train.train_steps
            start_step = global_step % args.train.train_steps
            lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
            environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])
            if start_step == 0:  # resume at the end of epoch
                iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    moe_gates = [
        m for m in model.modules() 
        if hasattr(m, "expert_bias") and hasattr(m, "routed_scaling_factor")
    ]

    if len(moe_gates) > 0:
        logger.info_rank0(f"Found {len(moe_gates)} active gates for Loss-Free Balancing.")
    else:
        logger.warning_rank0("WARNING: No MoE gates found! Check FSDP wrapping configurations.")
    # Create a dedicated directory to store local checkpoints of MoE metrics
    moe_monitor = MoEMonitor(
        model=model,
        num_experts=model_config.num_experts,
        num_layers=model_config.num_hidden_layers,
        first_k_dense_replace=model_config.first_k_dense_replace,
        phase_bins=args.train.phase_bins,
        log_dir=os.path.join(args.train.output_dir, "moe_logs")
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
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
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()
            num_accumulation_steps = len(micro_batches)
            total_consistency_loss = 0
            total_confidence_loss = 0

            expert_counts = None
            avg_max_vio = 0.0
            avg_bias_std = 0.0
            avg_bias_min = 0.0
            avg_bias_max = 0.0

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                if "mask_density" in micro_batch:
                    current_mask_density = micro_batch.pop("mask_density")
                else:
                    # Fallback to a zero-tensor matching the batch size
                    B_size = micro_batch["noisy_input_ids"].shape[0]
                    current_mask_density = torch.zeros(B_size, device=get_device_type())
                    logger.warning("Critical: Mask density not found. Defaulting to 0.0.")

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
                    
                    # --- DYNAMIC MASK CREATION ---
                    bd_attn_full_len = seq_len * 2
                    q_idx = torch.arange(bd_attn_full_len, device=get_device_type())[:, None]
                    kv_idx = torch.arange(bd_attn_full_len, device=get_device_type())[None, :]

                    attn_mask = torch.zeros((batch_size, 1, bd_attn_full_len, bd_attn_full_len), 
                                             dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16,
                                             device=get_device_type())

                    for b in range(batch_size):
                        b_seg = full_segment_ids[b] if full_segment_ids is not None else None
                        mask_flag = block_diffusion_mask(
                            q_idx=q_idx,
                            kv_idx=kv_idx,
                            block_size=args.train.block_size,
                            n=seq_len,
                            segment_ids=b_seg
                        )
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
                        
                        attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), 
                                                 dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16,
                                                 device=get_device_type())
                        for b in range(batch_size):
                            b_seg = segment_ids[b]
                            seg_q = b_seg[q_idx]
                            seg_kv = b_seg[kv_idx]
                            
                            # Must be in the same document, and not padding (-1)
                            same_doc = (seg_q == seg_kv) & (seg_q != -1) & (seg_kv != -1)
                            # Padding tokens only attend to themselves
                            is_pad = (seg_q == -1) & (q_idx == kv_idx)
                            
                            doc_mask = same_doc | is_pad
                            attn_mask[b, 0].masked_fill_(doc_mask.logical_not(), float("-inf"))
                        
                        micro_batch["attention_mask"] = attn_mask
                    else:
                        micro_batch["attention_mask"] = None

                labels = micro_batch.pop("labels", None)
                micro_batch.pop("mask_density", None)
                micro_batch.pop("noisy_input_ids", None) 

                phase_indices = micro_batch.pop("phase_indices", None)

                with model_fwd_context:
                    outputs = model(**micro_batch, use_cache=False, output_router_logits=True)
                    logits = outputs.logits
                    router_logits_tuple = outputs.router_logits # Tuple of (router_logits, topk_idx)
                    with torch.no_grad():
                        if expert_counts is None:
                            expert_counts = [
                                torch.zeros(model_config.num_experts, dtype=torch.long, device=get_device_type()) 
                                for _ in range(len(router_logits_tuple))
                            ]
                        
                        # Identify valid active tokens (filter dynamic padding)
                        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 156892
                        valid_mask = (micro_batch["input_ids"] != pad_id) # Shape: [batch_size, seq_len]
                        
                        for layer_idx, (_, topk_idx) in enumerate(router_logits_tuple):
                            # topk_idx: [batch_size, seq_len, top_k]
                            # Mask flattens topk_idx into a 2D matrix of valid tokens
                            valid_topk_idx = topk_idx[valid_mask] 
                            counts = torch.bincount(valid_topk_idx.flatten(), minlength=model_config.num_experts)
                            expert_counts[layer_idx] += counts
                    # -------------------------------------------------------------
                    if args.train.block_diffusion_mode:
                        noisy_logits = logits[:, :noisy_input_ids.shape[1]].contiguous()
                    else:
                        noisy_logits = logits

                    confidence_loss = torch.tensor(0.0, device=noisy_logits.device)
                    if args.train.confidence_beta > 0:
                        confidence_loss = compute_confidence_loss(
                            logits=noisy_logits,
                            labels=labels,
                        )

                    if args.train.same_token_labels:
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            noisy_logits.view(-1, noisy_logits.shape[-1]), 
                            labels.view(-1), 
                            reduction="none",
                        ).view(noisy_logits.shape[0], -1) 

                        valid_tokens = (labels != -100).sum() 
                        if valid_tokens == 0:
                            consistency_loss = unscaled_loss.sum() * 0.0
                        else:
                            consistency_loss = unscaled_loss.sum() / valid_tokens
                    else:
                        shifted_noisy_logits = noisy_logits[:, :-1, :].contiguous()
                        shifted_labels = labels[:, 1:].contiguous()
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            shifted_noisy_logits.view(-1, shifted_noisy_logits.shape[-1]),
                            shifted_labels.view(-1),
                            reduction="none",
                        ).view(shifted_noisy_logits.shape[0], -1)

                        valid_tokens = (shifted_labels != -100).sum()
                        if valid_tokens == 0:
                            # Fallback to prevent NaN. Multiply by 0 to keep the computation graph intact
                            consistency_loss = unscaled_loss.sum() * 0.0
                        else:
                            consistency_loss = unscaled_loss.sum() / valid_tokens
                        
                combined_loss = consistency_loss + confidence_loss * args.train.confidence_beta
                loss = combined_loss / num_accumulation_steps
                with model_bwd_context:
                    loss.backward()


                if phase_indices is not None:
                    if args.train.block_diffusion_mode:
                        # If BDLM, append Phase 0 for the clean context half
                        clean_phase = torch.zeros_like(clean_input_ids)
                        phase_indices = torch.cat([phase_indices, clean_phase], dim=1)

                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 156892
                    valid_mask = (micro_batch["input_ids"] != pad_id)

                    # Track Expert Activation occurrences in memory
                    moe_monitor.update_activations(
                        phase_indices=phase_indices, 
                        valid_mask=valid_mask, 
                        router_logits_tuple=router_logits_tuple
                    )

                total_loss += loss.item()
                total_consistency_loss += consistency_loss.item() / num_accumulation_steps
                total_confidence_loss += confidence_loss.item() / num_accumulation_steps
                del micro_batch

            # Track gradient norms before gradients are updated or cleared
            moe_monitor.update_gradients()

            # Apply Auxiliary-Loss-Free Bias Update & Calculate Stats
            if expert_counts is not None and len(moe_gates) > 0:
                with torch.no_grad():
                    # [num_layers, num_experts]
                    expert_counts_tensor = torch.stack(expert_counts).to(get_device_type())
                    
                    # 1. Synchronize token counts across EP/DP worlds
                    dist.all_reduce(expert_counts_tensor, op=dist.ReduceOp.SUM, group=None)
                    
                    update_rate = 0.001  # DeepSeek standard scale
                    total_max_vio = 0.0
                    total_bias_std = 0.0
                    global_bias_min = float('inf')
                    global_bias_max = float('-inf')
                    
                    # 2. Skew biases and collect metrics per layer
                    for i, gate in enumerate(moe_gates):
                        counts = expert_counts_tensor[i].float()
                        mean_count = counts.mean()
                        
                        # Load violation error
                        error = counts - mean_count
                        
                        if mean_count > 0:
                            layer_max_vio = (torch.abs(error) / mean_count).max().item()
                            total_max_vio += layer_max_vio
                        
                        # Update bias buffer in-place
                        gate.expert_bias.sub_(update_rate * torch.sign(error))        
                        gate.expert_bias.sub_(gate.expert_bias.mean())
                        
                        # Extract distribution metrics
                        total_bias_std += gate.expert_bias.std().item()
                        global_bias_min = min(global_bias_min, gate.expert_bias.min().item())
                        global_bias_max = max(global_bias_max, gate.expert_bias.max().item())
                    
                    # Average metrics across all MoE layers
                    avg_max_vio = total_max_vio / len(moe_gates)
                    avg_bias_std = total_bias_std / len(moe_gates)
            # ---------------------------------------------------------

            # Prefer model-provided clip_grad_norm_ (now both FSDP1 and FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                logger.info_rank0(
                    "Can NOT find registered clip_grad_norm_ method in the model, using PyTorch default implementation.."
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            LOG_INTERVAL = 20  
            MOE_SAVE_INTERVAL = 100

            if global_step % LOG_INTERVAL == 0:
                # Add telemetry to post-fix output
                data_loader_tqdm.set_postfix_str(
                    f"loss: {total_loss:.2f}, vio: {avg_max_vio:.3f}, bias_std: {avg_bias_std:.3f}"
                )

                if args.train.global_rank == 0:
                    if args.train.use_wandb:
                        train_metrics.update({
                            "training/loss": total_loss, 
                            "training/cons_loss": total_consistency_loss, 
                            "training/conf_loss": total_confidence_loss,  
                            "training/grad_norm": grad_norm, 
                            "training/lr": lr,
                            "training/noise_high": dynamic_noise_high.value,
                            # --- LOAD BALANCING METRICS ---
                            "moe/max_vio": avg_max_vio,
                            "moe/bias_std": avg_bias_std,
                            "moe/bias_min": global_bias_min,
                            "moe/bias_max": global_bias_max,
                        })
                        wandb.log(train_metrics, step=global_step)

                    # --- INJECTED SANITY CHECK PRINTER ---
                    sample_idx = 0
                    
                    # Labels are -100 for unmasked tokens. Find the actual masked targets.
                    masked_positions = (labels[sample_idx] != -100).nonzero(as_tuple=True)[0]
                    
                    mon_input_ids = noisy_input_ids 

                    if len(masked_positions) > 0:
                        logger.info(f"\n--- [TRAIN TEXT MONITOR] Step {global_step} ---")
                        
                        # 1. Convert logits to probabilities ONCE for the sample to get confidences
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

                            # 4. Slice from the NOISY sequence
                            left_ids = mon_input_ids[sample_idx, start_ctx:pos].tolist()
                            right_ids = mon_input_ids[sample_idx, pos+1:end_ctx].tolist()
                            
                            left_str = safe_decode_with_masks(left_ids)
                            right_str = safe_decode_with_masks(right_ids)
                            context_str = f"{left_str} [TARGET_MASK] {right_str}"
                            
                            target_id = labels[sample_idx, pos].item()
                            target_str = tokenizer.decode([target_id]).replace('\n', '\\n')
                            
                            # 5. Fetch Top-3 predictions to evaluate "meaningful alternatives"
                            topk_probs, topk_ids = torch.topk(probs[pos], k=3)
                            
                            logger.info(f"Context : ...{context_str.strip()}...")
                            logger.info(f"Target  : '{target_str}' (ID: {target_id})")
                            
                            # Build the prediction string with probabilities
                            pred_log = "Predict :"
                            for k in range(3):
                                p_id = topk_ids[k].item()
                                p_prob = topk_probs[k].item() * 100
                                p_str = tokenizer.decode([p_id]).replace('\n', '\\n')
                                marker = "✅" if p_id == target_id else "❌"
                                pred_log += f" | {k+1}. '{p_str}' ({p_prob:.1f}% {marker})"
                                
                            logger.info(pred_log)
                            logger.info("-" * 60)
                    # -------------------------------------

            if global_step % MOE_SAVE_INTERVAL == 0:
                moe_monitor.save(global_step)

            data_loader_tqdm.update()

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()