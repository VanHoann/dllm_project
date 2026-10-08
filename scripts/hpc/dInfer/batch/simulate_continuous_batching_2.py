import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import time
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

# Robust project root resolution (traverses upward to locate repository markers)
def find_project_root(current_path: Path) -> Path:
    for parent in current_path.resolve().parents:
        if (parent / "third_party").exists() or (parent / "setup.py").exists() or (parent / "README.md").exists():
            return parent
    return current_path.resolve().parents[3]

PROJECT_DIR = find_project_root(Path(__file__))
DINFER_PATH = PROJECT_DIR / "third_party" / "dInfer" / "python"
if DINFER_PATH.exists() and str(DINFER_PATH) not in sys.path:
    sys.path.insert(0, str(DINFER_PATH))

try:
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM, GLOBAL_TELEMETRY
except ImportError as e:
    raise ImportError(f"Could not import dInfer from {DINFER_PATH}. Ensure dependencies are installed: {e}")


class SequenceState:
    def __init__(self, seq_id, prompt_ids, target_gen_len, req_data, is_ghost=False, block_length=32, mask_id=156900, eos_ids=None):
        self.seq_id = seq_id
        self.req_data = req_data
        self.is_ghost = is_ghost
        
        self.prompt_len = prompt_ids.shape[1]
        self.block_length = block_length
        self.mask_id = mask_id
        self.eos_ids = eos_ids if eos_ids is not None else [156892]
        
        # Align target generation length to exact multiples of parallel block length
        gen_len_aligned = ((target_gen_len + block_length - 1) // block_length) * block_length
        self.total_length = self.prompt_len + gen_len_aligned

        # Initialize sequence buffer with prompt tokens followed by masks
        self.tokens = torch.full((1, self.total_length), mask_id, dtype=torch.long)
        self.tokens[0, :self.prompt_len] = prompt_ids[0]

        self.kv_cache = None
        self.cached_len = 0
        self.current_block_start = self.prompt_len
        
        self.state = "PREFILL_PROMPT" 
        self.current_phase = 0
        self.completion_time_steps = 0 
        self.wait_time = 0
        self.newly_accepted = torch.zeros(self.block_length, dtype=torch.bool)

    def _update_current_phase(self):
        end = self.current_block_start + self.block_length
        block_tokens = self.tokens[0, self.current_block_start:end]
        self.current_phase = int((block_tokens == self.mask_id).sum().item())
        
        if self.current_phase == 0:
            if any(eos in block_tokens for eos in self.eos_ids):
                self.state = "FINISHED"
                return
            
            if self.cached_len == 0:
                self.state = "PREFILL_PROMPT"
            else:
                self.state = "COMMIT_BLOCK"
        else:
            self.state = "DECODE"


class ContinuousBatchingEngine:
    def __init__(self, model, tokenizer, block_length=32, b_step=4, mask_id=156900, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.block_length = block_length
        self.b_step = b_step
        self.mask_id = mask_id
        
        eos_tokens = set()
        if hasattr(model.config, "eos_token_id") and model.config.eos_token_id is not None:
            if isinstance(model.config.eos_token_id, list):
                eos_tokens.update(model.config.eos_token_id)
            else:
                eos_tokens.add(model.config.eos_token_id)
        if hasattr(model.config, "role_end_token_id") and model.config.role_end_token_id is not None:
            eos_tokens.add(model.config.role_end_token_id)
        
        pad_id = getattr(model.config, "pad_token_id", 156892)
        if pad_id in eos_tokens and len(eos_tokens) > 1:
            eos_tokens.remove(pad_id)
            
        self.eos_ids = list(eos_tokens) if eos_tokens else [156895]
        self.device = device
        self.dense_layers = getattr(model.config, "first_k_dense_replace", 1)
        self.global_step = 0
        self.num_layers = model.config.num_hidden_layers

    def _format_kv(self, raw_kv):
        if isinstance(raw_kv, torch.Tensor) and raw_kv.shape[0] == self.num_layers:
            return tuple((raw_kv[l][0], raw_kv[l][1]) for l in range(self.num_layers))
        if isinstance(raw_kv, (list, tuple)) and len(raw_kv) == 2 * self.num_layers:
            return tuple((raw_kv[2 * l], raw_kv[2 * l + 1]) for l in range(self.num_layers))
        if isinstance(raw_kv, (list, tuple)) and len(raw_kv) == self.num_layers:
            return raw_kv
        raise ValueError("Unrecognized past_key_values format")

    def schedule(self, active_pool, policy="fcfs", base_wait_tolerance=5):
        t0 = time.perf_counter()
        
        queue_prefill = [s for s in active_pool if s.state == "PREFILL_PROMPT"]
        queue_commit = [s for s in active_pool if s.state == "COMMIT_BLOCK"]
        queue_decode = [s for s in active_pool if s.state == "DECODE"]

        batch, b_type = [], "IDLE"

        # 1. Strict Compute Priority: Prefill > Commit > Decode
        if queue_prefill:
            batch, b_type = queue_prefill[:1], "PREFILL_PROMPT"
        elif queue_commit:
            batch, b_type = queue_commit[:self.b_step], "COMMIT_BLOCK"
        elif queue_decode:
            b_type = "DECODE"
            if policy == "phase_aware":
                queue_decode.sort(key=lambda s: s.current_phase)
                
                # Dynamic starvation threshold
                dynamic_tolerance = max(base_wait_tolerance, (len(queue_decode) // self.b_step) * 2)
                starving_seqs = [s for s in queue_decode if getattr(s, 'wait_time', 0) > dynamic_tolerance]
                
                if starving_seqs:
                    if len(queue_decode) <= self.b_step:
                        batch = queue_decode
                    else:
                        target_seq = max(starving_seqs, key=lambda s: s.wait_time)
                        target_idx = queue_decode.index(target_seq)
                        
                        best_spread = float('inf')
                        best_start = 0
                        start_min = max(0, target_idx - self.b_step + 1)
                        start_max = min(len(queue_decode) - self.b_step, target_idx)
                        
                        for i in range(start_min, start_max + 1):
                            spread = queue_decode[i + self.b_step - 1].current_phase - queue_decode[i].current_phase
                            if spread < best_spread:
                                best_spread = spread
                                best_start = i
                        batch = queue_decode[best_start : best_start + self.b_step]
                else:
                    if len(queue_decode) <= self.b_step:
                        batch = queue_decode
                    else:
                        best_spread = float('inf')
                        best_idx = 0
                        for i in range(len(queue_decode) - self.b_step + 1):
                            spread = queue_decode[i + self.b_step - 1].current_phase - queue_decode[i].current_phase
                            if spread < best_spread:
                                best_spread = spread
                                best_idx = i
                        batch = queue_decode[best_idx : best_idx + self.b_step]
            else:
                batch = queue_decode[:self.b_step]
        
        # 2. Update queue wait times
        for s in active_pool:
            if s.state == "DECODE":
                if s not in batch:
                    s.wait_time += 1
                else:
                    s.wait_time = 0
        
        scheduling_time_ms = (time.perf_counter() - t0) * 1000
        return batch, b_type, scheduling_time_ms

    @torch.no_grad()
    def execute_compute_phase(self, batch_seqs):
        if "team_classes" in GLOBAL_TELEMETRY:
            GLOBAL_TELEMETRY["team_classes"] = None
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        bsz = len(batch_seqs)
        
        start_evt.record()

        if batch_seqs[0].state == "PREFILL_PROMPT":
            seq = batch_seqs[0]
            input_ids = seq.tokens[:, :seq.prompt_len].to(self.device)
            phase_indices = torch.zeros_like(input_ids) 
            pos_ids = torch.arange(0, seq.prompt_len, device=self.device).unsqueeze(0)
            
            # Construct block-causal attention mask for prefill
            num_blocks = (seq.prompt_len + self.block_length - 1) // self.block_length
            block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device, dtype=torch.bool))
            bd_attn_mask = block_mask.repeat_interleave(self.block_length, dim=0).repeat_interleave(self.block_length, dim=1)
            bd_attn_mask = bd_attn_mask[:seq.prompt_len, :seq.prompt_len].unsqueeze(0).unsqueeze(0)
            
            out = self.model(
                input_ids=input_ids, 
                position_ids=pos_ids, 
                phase_indices=phase_indices, 
                attention_mask=bd_attn_mask, 
                use_cache=True
            )
            seq.kv_cache = self._format_kv(out.past_key_values)
            seq.cached_len = seq.prompt_len
            seq._update_current_phase()

        else:
            max_cached_len = max([s.cached_len for s in batch_seqs])
            target_cache_len = max_cached_len + self.block_length
            
            padded_kv = []
            for l in range(self.num_layers):
                k_list, v_list = [], []
                for s in batch_seqs:
                    k, v = s.kv_cache[l]
                    pad_len = target_cache_len - k.shape[2]
                    if pad_len > 0:
                        k_pad = torch.zeros((1, k.shape[1], pad_len, k.shape[3]), device=self.device, dtype=k.dtype)
                        v_pad = torch.zeros((1, v.shape[1], pad_len, v.shape[3]), device=self.device, dtype=v.dtype)
                        k_list.append(torch.cat([k, k_pad], dim=2))
                        v_list.append(torch.cat([v, v_pad], dim=2))
                    else:
                        k_list.append(k)
                        v_list.append(v)
                padded_kv.append((torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)))
            padded_kv = tuple(padded_kv)

            batch_tokens = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)
            pos_ids = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)
            attn_mask = torch.zeros((bsz, 1, self.block_length, max_cached_len + self.block_length), dtype=torch.bool, device=self.device)
            phase_indices = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)

            for i, s in enumerate(batch_seqs):
                batch_tokens[i, :] = s.tokens[0, s.current_block_start:s.current_block_start + self.block_length]
                pos_ids[i, :] = torch.arange(s.cached_len, s.cached_len + self.block_length, device=self.device)
                attn_mask[i, 0, :, :s.cached_len] = True              
                attn_mask[i, 0, :, max_cached_len:] = True

            out = self.model(
                input_ids=batch_tokens, position_ids=pos_ids, phase_indices=phase_indices,
                attention_mask=attn_mask, past_key_values=padded_kv, use_cache=True 
            )

            formatted_out = self._format_kv(out.past_key_values)
            for i, s in enumerate(batch_seqs):
                new_kv = []
                for l in range(self.num_layers):
                    k_full = formatted_out[l][0][i:i+1]
                    v_full = formatted_out[l][1][i:i+1]
                    k_valid = torch.cat([k_full[:, :, :s.cached_len, :], k_full[:, :, max_cached_len:, :]], dim=2)
                    v_valid = torch.cat([v_full[:, :, :s.cached_len, :], v_full[:, :, max_cached_len:, :]], dim=2)
                    new_kv.append((k_valid, v_valid))
                    
                s.kv_cache = tuple(new_kv)
                s.cached_len += self.block_length
                s.current_block_start += self.block_length
                
                if s.current_block_start >= s.total_length:
                    s.state = "FINISHED"
                else:
                    s._update_current_phase()
            
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt)

    @torch.no_grad()
    def execute_decode_batch(self, batch_seqs):
        bsz = len(batch_seqs)
        max_cached_len = max([s.cached_len for s in batch_seqs])
        target_cache_len = max_cached_len + self.block_length
        
        padded_kv = []
        for l in range(self.num_layers):
            k_list, v_list = [], []
            for s in batch_seqs:
                k, v = s.kv_cache[l]
                pad_len = target_cache_len - k.shape[2]
                if pad_len > 0:
                    k_pad = torch.zeros((1, k.shape[1], pad_len, k.shape[3]), device=self.device, dtype=k.dtype)
                    v_pad = torch.zeros((1, v.shape[1], pad_len, v.shape[3]), device=self.device, dtype=v.dtype)
                    k_list.append(torch.cat([k, k_pad], dim=2))
                    v_list.append(torch.cat([v, v_pad], dim=2))
                else:
                    k_list.append(k)
                    v_list.append(v)
            padded_kv.append((torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)))
        padded_kv = tuple(padded_kv)

        batch_tokens = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)
        phase_indices = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)
        pos_ids = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)
        
        for i, s in enumerate(batch_seqs):
            start = s.current_block_start
            batch_tokens[i, :] = s.tokens[0, start:start + self.block_length]
            phase_indices[i, :] = s.current_phase
            pos_ids[i, :] = torch.arange(s.cached_len, s.cached_len + self.block_length, device=self.device)

        attn_mask = torch.zeros((bsz, 1, self.block_length, max_cached_len + self.block_length), dtype=torch.bool, device=self.device)
        for i, s in enumerate(batch_seqs):
            attn_mask[i, 0, :, :s.cached_len] = True              
            attn_mask[i, 0, :, max_cached_len:] = True            

        # Dynamic Token Classification for Spatio-temporal Competitors
        if GLOBAL_TELEMETRY.get("competitor_method") == "team":
            team_classes = torch.zeros((bsz, self.block_length), dtype=torch.long, device=self.device)
            for i, s in enumerate(batch_seqs):
                start = s.current_block_start
                block_tokens = s.tokens[0, start:start + self.block_length].to(self.device)
                is_mask = (block_tokens == self.mask_id)
                is_decoded = ~is_mask
                
                hot_mask = torch.zeros_like(is_mask)
                decoded_idx = is_decoded.nonzero(as_tuple=True)[0]
                mask_idx = is_mask.nonzero(as_tuple=True)[0]
                
                if len(decoded_idx) > 0 and len(mask_idx) > 0:
                    dist_matrix = torch.abs(mask_idx.unsqueeze(1) - decoded_idx.unsqueeze(0))
                    min_dist = torch.min(dist_matrix, dim=1).values
                    hot_mask_indices = mask_idx[min_dist <= 3]
                    hot_mask[hot_mask_indices] = True
                    
                newly_accepted = s.newly_accepted.to(self.device)
                
                class1 = hot_mask | newly_accepted
                class2 = is_mask & ~hot_mask
                class0 = is_decoded & ~newly_accepted
                
                team_classes[i, class1] = 1
                team_classes[i, class2] = 2
                team_classes[i, class0] = 0
                
                s.newly_accepted.zero_()
                
            GLOBAL_TELEMETRY["team_classes"] = team_classes

        if GLOBAL_TELEMETRY.get("step_unique_experts") is not None:
            GLOBAL_TELEMETRY["step_unique_experts"].fill_(0)
        GLOBAL_TELEMETRY["moe_events"].clear()
        
        if "des_overhead_events" in GLOBAL_TELEMETRY:
            GLOBAL_TELEMETRY["des_overhead_events"].clear()

        if "team_overhead_events" in GLOBAL_TELEMETRY:
            GLOBAL_TELEMETRY["team_overhead_events"].clear()

        if "dmoe_overhead_events" in GLOBAL_TELEMETRY:
            GLOBAL_TELEMETRY["dmoe_overhead_events"].clear()
        
        torch.cuda.synchronize()
        output = self.model(
            input_ids=batch_tokens, position_ids=pos_ids, phase_indices=phase_indices,
            attention_mask=attn_mask, past_key_values=padded_kv, 
            use_cache=False 
        )
        torch.cuda.synchronize()

        total_moe_ms = sum([s_evt.elapsed_time(e_evt) for s_evt, e_evt, _ in GLOBAL_TELEMETRY["moe_events"]])
        
        des_overhead_ms = sum([s_evt.elapsed_time(e_evt) for s_evt, e_evt in GLOBAL_TELEMETRY.get("des_overhead_events", [])])
        team_overhead_ms = sum([s_evt.elapsed_time(e_evt) for s_evt, e_evt in GLOBAL_TELEMETRY.get("team_overhead_events", [])])
        dmoe_overhead_ms = sum([s_evt.elapsed_time(e_evt) for s_evt, e_evt in GLOBAL_TELEMETRY.get("dmoe_overhead_events", [])])

        layerwise_unique = []
        if GLOBAL_TELEMETRY.get("step_unique_experts") is not None:
            layerwise_unique = GLOBAL_TELEMETRY["step_unique_experts"][self.dense_layers:].cpu().tolist()
        mean_unique_experts = float(np.mean(layerwise_unique)) if layerwise_unique else 0.0

        phase_spread = max(s.current_phase for s in batch_seqs) - min(s.current_phase for s in batch_seqs)

        logits = output.logits
        probs = F.softmax(logits, dim=-1)
        max_probs, predicted_ids = torch.max(probs, dim=-1)
        
        max_probs = max_probs.cpu()
        predicted_ids = predicted_ids.cpu()

        target_threshold = 0.97 
        
        for idx, seq in enumerate(batch_seqs):
            start = seq.current_block_start
            end = start + self.block_length
            
            block_tokens = seq.tokens[0, start:end]
            mask_positions = (block_tokens == self.mask_id).nonzero(as_tuple=True)[0]
            
            if len(mask_positions) == 0:
                continue 
            
            mask_confidences = max_probs[idx, mask_positions]
            mask_predictions = predicted_ids[idx, mask_positions]
            
            valid_candidates = mask_predictions != self.mask_id
            if not valid_candidates.any():
                continue 
                
            valid_positions = mask_positions[valid_candidates]
            valid_confidences = mask_confidences[valid_candidates]
            valid_predictions = mask_predictions[valid_candidates]
            
            max_conf = torch.max(valid_confidences)
            actual_threshold = torch.clamp(max_conf - 1e-5, max=target_threshold)
            
            accepted_mask = valid_confidences >= actual_threshold
            accepted_positions = valid_positions[accepted_mask]
            block_tokens[accepted_positions] = valid_predictions[accepted_mask]
            
            if GLOBAL_TELEMETRY.get("competitor_method") == "team":
                seq.newly_accepted[accepted_positions] = True

            seq.tokens[0, start:end] = block_tokens
            seq._update_current_phase()

        return {
            "unique_experts": mean_unique_experts,
            "fwd_time_ms": total_moe_ms,
            "des_overhead_ms": des_overhead_ms,
            "team_overhead_ms": team_overhead_ms,
            "dmoe_overhead_ms": dmoe_overhead_ms,
            "phase_spread": phase_spread,
            "layerwise_unique": layerwise_unique
        }


def run_simulation(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    import socket
    os.environ['MASTER_ADDR'] = 'localhost'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        free_port = s.getsockname()[1]
        
    os.environ['MASTER_PORT'] = str(free_port)    
    from sglang.srt import distributed
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1, backend='nccl')
    
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2Gate
    def new_forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
        return logits.to(torch.float32)
    LLaDA2Gate.forward = new_forward
    
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    config.n_group = 0
    config.topk_group = 0
    GLOBAL_TELEMETRY["profile_moe_kernel"] = True
    GLOBAL_TELEMETRY["measure_batch_unique_experts"] = True 

    GLOBAL_TELEMETRY["competitor_method"] = args.competitor_method
    if args.competitor_method == "des_vote":
        GLOBAL_TELEMETRY["des_m_core"] = args.des_m_core
        GLOBAL_TELEMETRY["des_mode"] = args.des_mode
        GLOBAL_TELEMETRY["phase_moe_mode"] = "none" 
    elif args.competitor_method == "team":
        GLOBAL_TELEMETRY["team_lac_mode"] = args.team_lac_mode
        GLOBAL_TELEMETRY["phase_moe_mode"] = "none" 
    elif args.competitor_method == "dmoe":
        GLOBAL_TELEMETRY["dmoe_p"] = args.dmoe_p
        GLOBAL_TELEMETRY["dmoe_guard_size"] = args.dmoe_guard_size
        GLOBAL_TELEMETRY["dmoe_mode"] = args.dmoe_mode
        GLOBAL_TELEMETRY["phase_moe_mode"] = "none"
    else:
        GLOBAL_TELEMETRY["phase_moe_mode"] = getattr(config, "phase_moe_mode", "none")

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        set_global_server_args_for_scheduler = None

    try:
        from sglang.srt.layers.moe import initialize_moe_config
    except ImportError:
        def initialize_moe_config(*args, **kwargs): pass
        
    server_args = ServerArgs(
        model_path=args.model_path, 
        enable_dp_attention=True, 
        trust_remote_code=True, 
        tp_size=1, 
        dp_size=1, 
        pp_size=1
    )
    
    if set_global_server_args_for_scheduler is not None:
        set_global_server_args_for_scheduler(server_args)

    initialize_dp_attention(server_args=server_args, model_config=config)
    initialize_moe_config(server_args)

    model = LLaDA2SGLangLM(config=config, expert_map_path=".").eval()
    model.load_weights(args.model_path, torch_dtype=torch.bfloat16, device=device)
    model = model.to(dtype=torch.bfloat16, device=device)

    for m in model.modules():
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            if isinstance(m.cos_sin_cache, torch.nn.Parameter):
                m.cos_sin_cache.data = m.cos_sin_cache.data.to(dtype=torch.float32, device=device)
            else:
                m.cos_sin_cache = m.cos_sin_cache.to(dtype=torch.float32, device=device)
        
        for bias_attr in ["e_score_correction_bias", "expert_bias", "correction_bias"]:
            if hasattr(m, bias_attr) and getattr(m, bias_attr) is not None:
                b = getattr(m, bias_attr)
                if isinstance(b, torch.nn.Parameter):
                    b.data = b.data.to(dtype=torch.float32, device=device)
                else:
                    setattr(m, bias_attr, b.to(dtype=torch.float32, device=device))

    with open(args.workload_path, "r", encoding="utf-8") as f:
        workload_data = [json.loads(line) for line in f]
    
    total_real_requests = len(workload_data)
    print(f"Loaded {total_real_requests} requests from {os.path.basename(args.workload_path)}.")
    print(f"Mode: {args.scheduler_type.upper()} | C={args.concurrency} | B_step={args.b_step}")

    engine = ContinuousBatchingEngine(
        model=model, tokenizer=tokenizer, block_length=args.block_length,
        b_step=args.b_step, mask_id=getattr(config, "mask_token_id", 156900), device=device
    )

    print("Warming up CUDA kernels...")
    dummy_input = torch.zeros((1, args.block_length), dtype=torch.long, device=device)
    _ = model(input_ids=dummy_input, position_ids=dummy_input, phase_indices=dummy_input)
    torch.cuda.synchronize()

    active_pool = []
    real_requests_injected = 0
    real_requests_finished = 0
    ghost_counter = 0
    
    for _ in range(min(args.concurrency, total_real_requests)):
        req = workload_data[real_requests_injected]
        input_ids = tokenizer(req["prompt"], return_tensors="pt")["input_ids"]
        seq = SequenceState(
            seq_id=real_requests_injected, 
            prompt_ids=input_ids, 
            target_gen_len=req["gen_length"], 
            req_data=req,
            is_ghost=False,
            block_length=args.block_length, 
            mask_id=engine.mask_id,
            eos_ids=engine.eos_ids
        )
        active_pool.append(seq)
        real_requests_injected += 1

    step_unique, step_latency, step_spread = [], [], []
    step_prefill_time, step_commit_time, step_sched_time = [], [], []
    step_des_overhead = []
    step_team_overhead = []
    step_dmoe_overhead = []
    run_ttc = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output_predictions)), exist_ok=True)
    engine.global_step = 0
    start_time = time.time()

    # Context manager to ensure safe, flushed writes even on interruption
    with open(args.output_predictions, "w", encoding="utf-8") as pred_file:
        while real_requests_finished < total_real_requests:
            batch, batch_type, sched_overhead = engine.schedule(active_pool, policy=args.scheduler_type)

            if not batch:
                time.sleep(0.05)
                continue

            if batch_type in ["PREFILL_PROMPT", "COMMIT_BLOCK"]:
                p0_latency = engine.execute_compute_phase(batch)
                if batch_type == "PREFILL_PROMPT":
                    step_prefill_time.append(p0_latency)
                else:
                    step_commit_time.append(p0_latency)
                    
            elif batch_type == "DECODE":
                stats = engine.execute_decode_batch(batch)
                step_unique.append(stats["unique_experts"])
                step_latency.append(stats["fwd_time_ms"])
                step_spread.append(stats["phase_spread"])
                step_des_overhead.append(stats.get("des_overhead_ms", 0.0))
                step_team_overhead.append(stats.get("team_overhead_ms", 0.0))
                step_dmoe_overhead.append(stats.get("dmoe_overhead_ms", 0.0))
            
            step_sched_time.append(sched_overhead)
            engine.global_step += 1
            
            finished_seqs = [s for s in active_pool if s.state == "FINISHED"]
            
            for s in finished_seqs:
                if not s.is_ghost:
                    gen_tokens_raw = s.tokens[0, s.prompt_len:].cpu()
                    eos_indices = []
                    for eos_id in engine.eos_ids:
                        eos_indices.extend((gen_tokens_raw == eos_id).nonzero(as_tuple=True)[0].tolist())
                    
                    if eos_indices:
                        first_eos_idx = min(eos_indices)
                        gen_tokens_raw = gen_tokens_raw[:first_eos_idx]

                    gen_tokens = gen_tokens_raw[gen_tokens_raw != engine.mask_id]
                    pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                    
                    out_item = {
                        "task": s.req_data["task"],
                        "task_type": s.req_data["task_type"],
                        "prompt": s.req_data["prompt"],
                        "ground_truth": s.req_data["ground_truth"],
                        "prediction": pred_text,
                        "raw_data": s.req_data["raw_data"]
                    }
                    pred_file.write(json.dumps(out_item) + "\n")
                    pred_file.flush()
                    
                    run_ttc.append(s.completion_time_steps)
                    real_requests_finished += 1
                    
                    if real_requests_finished % 10 == 0 or real_requests_finished == total_real_requests:
                        elapsed = time.time() - start_time
                        avg_apf = np.mean(step_unique[-50:]) if step_unique else 0.0
                        avg_spread = np.mean(step_spread[-50:]) if step_spread else 0.0
                        print(f"Progress: {real_requests_finished}/{total_real_requests} real sequences finished. "
                              f"Elapsed: {elapsed:.2f}s | Avg APF: {avg_apf:.1f} | Avg Spread: {avg_spread:.1f}")
                
                active_pool.remove(s)
                
                if real_requests_finished < total_real_requests: 
                    if real_requests_injected < total_real_requests:
                        next_req = workload_data[real_requests_injected]
                        is_ghost = False
                        next_seq_id = real_requests_injected
                        real_requests_injected += 1
                    else:
                        next_req = random.choice(workload_data)
                        is_ghost = True
                        ghost_counter -= 1
                        next_seq_id = ghost_counter
                    
                    input_ids = tokenizer(next_req["prompt"], return_tensors="pt")["input_ids"]
                    new_seq = SequenceState(
                        seq_id=next_seq_id, 
                        prompt_ids=input_ids, 
                        target_gen_len=next_req["gen_length"], 
                        req_data=next_req,
                        is_ghost=is_ghost,
                        block_length=args.block_length, 
                        mask_id=engine.mask_id,
                        eos_ids=engine.eos_ids
                    )
                    active_pool.append(new_seq)
            
            for s in active_pool:
                s.completion_time_steps += 1

    # Anonymized summary metrics: only stores basename for model/workload to prevent absolute path leaks
    results = {
        "model_name": os.path.basename(os.path.normpath(args.model_path)),
        "scheduler_type": args.scheduler_type,
        "concurrency": args.concurrency,
        "b_step": args.b_step,
        "unique_experts_mean": round(float(np.mean(step_unique)), 2) if step_unique else 0.0,
        "unique_experts_std": round(float(np.std(step_unique)), 2) if step_unique else 0.0,
        "moe_latency_ms_mean": round(float(np.mean(step_latency)), 2) if step_latency else 0.0,
        "moe_latency_ms_std": round(float(np.std(step_latency)), 2) if step_latency else 0.0,
        "phase_spread_mean": round(float(np.mean(step_spread)), 2) if step_spread else 0.0,
        "phase_spread_std": round(float(np.std(step_spread)), 2) if step_spread else 0.0,
        "completion_time_steps_mean": round(float(np.mean(run_ttc)), 2) if run_ttc else 0.0,
        "completion_time_steps_std": round(float(np.std(run_ttc)), 2) if run_ttc else 0.0,
        "completion_time_steps_p95": round(float(np.percentile(run_ttc, 95)), 2) if run_ttc else 0.0, 
        "completion_time_steps_p99": round(float(np.percentile(run_ttc, 99)), 2) if run_ttc else 0.0,
        "overhead_prefill_ms": round(float(np.mean(step_prefill_time)), 2) if step_prefill_time else 0.0,
        "overhead_des_ms": round(float(np.mean(step_des_overhead)), 4) if step_des_overhead else 0.0,
        "overhead_team_ms": round(float(np.mean(step_team_overhead)), 4) if step_team_overhead else 0.0,
        "overhead_dmoe_ms": round(float(np.mean(step_dmoe_overhead)), 4) if step_dmoe_overhead else 0.0,
        "overhead_commit_ms": round(float(np.mean(step_commit_time)), 2) if step_commit_time else 0.0,
        "overhead_sched_ms": round(float(np.mean(step_sched_time)), 4) if step_sched_time else 0.0,
        "total_run_time_sec": round(time.time() - start_time, 2)
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_hardware_json)), exist_ok=True)
    with open(args.output_hardware_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Run Completed: [{args.scheduler_type.upper()} | C={args.concurrency} | B_step={args.b_step}]")
    print(f"  -> Real requests finished: {total_real_requests}")
    print(f"  -> Predictions saved to  : {args.output_predictions}")
    print(f"  -> Telemetry saved to    : {args.output_hardware_json}")
    print(f"  -> Unique Experts Union  : {results['unique_experts_mean']:.2f} ± {results['unique_experts_std']:.2f}")
    print(f"  -> Average Phase Spread  : {results['phase_spread_mean']:.2f}")
    print(f"  -> Decode MoE Latency    : {results['moe_latency_ms_mean']:.2f} ms")
    print(f"  -> Avg Completion Steps  : {results['completion_time_steps_mean']:.2f}")
    print("=" * 60 + "\n")
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Continuous Batching for MoE Block Diffusion Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--workload_path", type=str, default="mixed_poc_workload_2.jsonl", help="Path to evaluation JSONL workload")
    parser.add_argument("--output_predictions", type=str, required=True, help="Path to output predictions JSONL")
    parser.add_argument("--output_hardware_json", type=str, required=True, help="Path to output hardware telemetry JSON")
    parser.add_argument("--scheduler_type", type=str, choices=["fcfs", "phase_aware"], required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--b_step", type=int, default=4)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--competitor_method", type=str, default="none", choices=["none", "des_vote", "team", "dmoe"])
    parser.add_argument("--des_m_core", type=int, default=96, help="Coreset budget factor for DES")
    parser.add_argument("--des_mode", type=str, default="per_sequence", choices=["per_sequence", "whole_batch", "none"])
    parser.add_argument("--team_lac_mode", type=str, default="lac_seq", choices=["lac_seq", "lac_batch", "none"])
    parser.add_argument("--dmoe_p", type=float, default=0.6, help="Top-P threshold for dMoE")
    parser.add_argument("--dmoe_guard_size", type=int, default=16, help="Minimum experts per block for dMoE")
    parser.add_argument("--dmoe_mode", type=str, default="per_sequence", choices=["per_sequence", "whole_batch", "none"])
    args = parser.parse_args()

    run_simulation(args)