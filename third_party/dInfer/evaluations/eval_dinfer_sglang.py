import accelerate
import torch
import random
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm, trange
import accelerate
import random
import numpy as np
import json
import time
import datasets
import json
import math
import datasets
import os
from transformers import AutoTokenizer, AutoConfig
import torch.multiprocessing as mp
from multiprocessing import Process
from lm_eval.api.model import LM
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer.model import LLaDAMoeModelLM, LLaDAModelLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockDiffusionLLM    
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.moe import initialize_moe_config
from dataclasses import dataclass


datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
datasets.config.DOWNLOAD_TIMEOUT = 180 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


bucket_size = 32
used_buckets = []

def cut_eos(data, eos_id=156892):
    eos_indices = (data[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        first_eos_idx = eos_indices[0].item()
        return data[:, :first_eos_idx]
    else:
        return data

@ torch.no_grad()
def run_benchmark(world_size, rank, gpu_id, tokenizer, args):
    print('started', world_size, rank, gpu_id)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    # --- 1. FORCE BFLOAT16 DISPATCH ---
    torch.set_default_dtype(torch.bfloat16)

    all_input_ids, padded_gen_lens = args.all_input_ids, args.padded_gen_lens

    block_length = args.block_length
    mask_id = args.mask_id
    eos_id = args.eos_id

    from sglang.srt import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')    
    # Fix: Explicitly define TP, PP, and EP to match dFactory training
    distributed.initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.tp_size,
        backend='nccl'
    )
    #distributed.initialize_model_parallel(args.tp_size, args.tp_size, 1, backend='nccl')
    print("[Loading model]")

    from sglang.srt.layers.dp_attention import initialize_dp_attention
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    # Force standard Top-K routing to match the dFactory training configuration
    model_config.n_group = 0
    model_config.topk_group = 0
    args.vocab_size = model_config.vocab_size
    print(f"[INIT] active phase_moe_mode: {getattr(model_config, 'phase_moe_mode', 'none')}", flush=True)
    server_args = ServerArgs(model_path=args.model_name, enable_dp_attention=True, trust_remote_code=True, tp_size=args.tp_size, dp_size=1, pp_size=1)
    
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server_args)
        
    initialize_dp_attention(
        server_args=server_args,
        model_config=model_config,
    )
    initialize_moe_config(server_args)
    
    # --- 2. LOAD WEIGHTS INTO THE SKELETON ---
    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    
    print(f"[Loading weights from {args.model_name}]")
    model.load_weights(args.model_name, torch_dtype=torch.bfloat16, device=device)
    model = model.to(dtype=torch.bfloat16, device=device)
    
    # --- 3. ROBUST SGLANG FIX ---
    # SGLang's `moe_fused_gate` requires the logits (input) and correction bias to have the exact same dtype.
    # Routing should be in float32 for numerical stability.

    # 1. Patch the Gate's forward to always cast logits to float32
    import torch.nn.functional as F
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2Gate
    
    old_forward = LLaDA2Gate.forward
    def new_forward(self, hidden_states):
        # Calculate logits in the model's native dtype (bfloat16)
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
        # Force float32 to match SGLang's expert_bias expectations
        return logits.to(torch.float32) 
        
    LLaDA2Gate.forward = new_forward

    # 2. Iterate through modules to fix cache and biases
    for m in model.modules():
        # SGLang's RoPE kernel strictly demands Float32 (not bfloat16!)
        if hasattr(m, "cos_sin_cache") and m.cos_sin_cache is not None:
            if isinstance(m.cos_sin_cache, torch.nn.Parameter):
                m.cos_sin_cache.data = m.cos_sin_cache.data.to(dtype=torch.float32, device=device)
            else:
                m.cos_sin_cache = m.cos_sin_cache.to(dtype=torch.float32, device=device)
        
        # Ensure all MoE biases are correctly shaped as float32 to match the patched logits
        for bias_attr in ["e_score_correction_bias", "expert_bias", "correction_bias"]:
            if hasattr(m, bias_attr) and getattr(m, bias_attr) is not None:
                b = getattr(m, bias_attr)
                if isinstance(b, torch.nn.Parameter):
                    b.data = b.data.to(dtype=torch.float32, device=device)
                else:
                    setattr(m, bias_attr, b.to(dtype=torch.float32, device=device))
    # ------------------------------------------------

    # --- 4. PREVENT NCCL DEADLOCKS FROM FP NON-DETERMINISM ---
    # Intercept the forward pass of the base SGLang wrapper to broadcast 
    # the final vocab logits from Rank 0 to all other ranks.
    old_forward = model.forward
    
    def synced_forward(*args, **kwargs):
        outputs = old_forward(*args, **kwargs)
        
        # Force bit-level agreement on logits across all TP ranks
        import sglang.srt.distributed as sglang_dist
        if sglang_dist.get_tensor_model_parallel_world_size() > 1:
            tp_group_wrapper = sglang_dist.get_tensor_model_parallel_group()
            
            # Extract the native PyTorch ProcessGroup from SGLang's wrapper
            native_pg = tp_group_wrapper.device_group if hasattr(tp_group_wrapper, "device_group") else tp_group_wrapper
            
            torch.distributed.broadcast(
                outputs.logits, 
                src=0, 
                group=native_pg
            )
        return outputs
        
    model.forward = synced_forward
    # ---------------------------------------------------------
    initialize_moe_config(server_args)

    input_lengths = [inp.size(-1) for inp in all_input_ids]
    max_length = max(input_lengths) + args.gen_len
    
    # ModelRunner now wraps a fully populated, uniformly-typed model
    supported_bs = [2**i for i in range(int(math.log2(args.batch_size)) + 1)]
    if args.batch_size not in supported_bs:
        supported_bs.append(args.batch_size)
    supported_bs = sorted(list(set(supported_bs)))
    model = ModelRunner(model, device, server_args=server_args, max_length=max_length, supported_batch_sizes=supported_bs, enable_cuda_graph=args.use_cudagraph, enable_compile=args.use_compile)
    
    batch_size = args.batch_size
    
    if args.parallel_decoding == 'threshold':
        if args.use_credit:
            decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)
        else:
            decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)

    else:
        decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=mask_id, eos_id=eos_id)
    decoder.tokenizer = tokenizer #Attach tokenizer for trajectory logging
    use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0

    if args.cache == 'prefix' or args.cache == 'dual':
        cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd, backend='sglang', max_length=max_length)
        # cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd)

    else:
        cache_factory=None

    if not args.use_bd:
        if args.cont_weight>0:
            if use_sw:
                print("IterSmoothWithVicinityCacheDiffusionLLM")
                dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                    cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                print("IterSmoothDiffusionLLM")
                dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
        else:
            if use_sw:
                print("VicinityCacheDiffusionLLM")
                dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                print("BlockWiseDiffusionLLM")
                dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=args.use_shift)
    else:
        print("BlockDiffusionLLM")
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), cache_factory=cache_factory, early_stop=True, maximum_unroll=4, expected_tpf=4, backend='sglang', use_shift=args.use_shift, use_naive_batching=False, mini_batch_size=args.batch_size)
        # dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True)

    
            
    input_lengths = [inp.size(-1) for inp in all_input_ids]
    sorted_indices = sorted(range(len(input_lengths)), key=lambda i: input_lengths[i])

    sorted_input_ids = [all_input_ids[i] for i in sorted_indices]
    sorted_padded_gen_lens = [padded_gen_lens[i] for i in sorted_indices]

    for wi in range(1):
        outputs = []
        total_forward = 0
        if rank==0:
            iterator = trange(0, len(sorted_input_ids), batch_size)
        else:
            iterator = range(0, len(sorted_input_ids), batch_size)
        start = time.time()
        tpfs = []
        tpss = []
        fpss = []
        total_token = 0
        token_numbers = []
        total_time = 0
        for i in iterator:   
            input_ids = sorted_input_ids[i:i+batch_size]

            prefill_blocks = input_ids[-1].shape[1] // block_length
            prefill_length = prefill_blocks * block_length

            max_length = input_ids[-1].shape[1]
            min_padded_length = sorted_padded_gen_lens[i+len(input_ids)-1]
            batch_input_ids= torch.zeros((len(input_ids), max_length), dtype=torch.long, device=device).fill_(mask_id)
            for j in range(len(input_ids)):
                batch_input_ids[j, :input_ids[j].shape[1]] = input_ids[j].to(device)
            input_ids = batch_input_ids
            inner_start = time.time()
            prev_forwards = dllm.num_forwards
            out = dllm.generate(input_ids, gen_length=min_padded_length, block_length=block_length)
            nfe = dllm.num_forwards - prev_forwards
            inner_stop = time.time()
            sample_time = inner_stop - inner_start

            # ============================================================
            # RESOLVE MoE KERNEL TIMES 
            # ============================================================
            import dinfer.model.modeling_llada2_moe_sglang as sglang_model
            if getattr(sglang_model, "GLOBAL_TELEMETRY", {}).get("profile_moe_kernel"):
                torch.cuda.synchronize()
                step_moe_decode = 0.0
                step_moe_prefill = 0.0
                for start_evt, end_evt, is_dec in sglang_model.GLOBAL_TELEMETRY["moe_events"]:
                    t = start_evt.elapsed_time(end_evt)
                    if is_dec:
                        step_moe_decode += t
                    else:
                        step_moe_prefill += t
                sglang_model.GLOBAL_TELEMETRY["moe_events"].clear()
                sglang_model.GLOBAL_TELEMETRY["total_moe_time_ms_decode"] += step_moe_decode
                sglang_model.GLOBAL_TELEMETRY["total_moe_time_ms_prefill"] += step_moe_prefill
            # ============================================================
            # ========================================================================
            # CODE INJECTION: LIVE STEP MONITORING
            # ========================================================================
            if rank == 0:
                for j in range(len(input_ids)):
                    try:
                        # Extract and decode the original prompt tokens
                        prompt_tokens = sorted_input_ids[i+j][0]
                        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
                        
                        # Extract and decode what the model predicted
                        prompt_len = sorted_input_ids[i+j].shape[1]
                        pred_tokens_raw = out[j, prompt_len:].unsqueeze(0)
                        answer_tokens = cut_eos(pred_tokens_raw, eos_id=-1)[0]
                        pred_text = tokenizer.decode(answer_tokens, skip_special_tokens=False)
                        
                        # --- FIX: Clean up the prompt tail ---
                        target_query_marker = "<role>HUMAN</role>"
                        if target_query_marker in prompt_text:
                            isolated_prompt = target_query_marker + prompt_text.split(target_query_marker)[-1]
                        else:
                            isolated_prompt = prompt_text
                        
                        # Calculate total sequence length (Prompt + Generated)
                        generated_len = answer_tokens.shape[0]
                        total_seq_len = prompt_len + generated_len

                        print(f"\n=== [MONITOR] Global Step: {i} | Batch Index: {j} ===")
                        print(f"--- LENGTHS --- Prompt: {prompt_len} | Generated: {generated_len} | Total: {total_seq_len}/2048")
                        print(f"--- TARGET PROMPT ---\n{isolated_prompt.strip()}")
                        print(f"--- MODEL PREDICTION ---\n{pred_text.strip()}")
                        print(f"==================================================\n", flush=True)
                    except Exception as log_e:
                        print(f"[MONITOR ERROR] Failed to log sample {j} at step {i}: {log_e}", flush=True)
            # ========================================================================
            
            for j in range(input_ids.shape[0]):
                outputs.append(out[j].unsqueeze(0))
            total_forward += nfe
            total_time += sample_time
            batch_token_number = 0
            
            # Keep track of prompt lengths for the batch summary
            batch_prompt_lengths = []
            
            for j in range(input_ids.shape[0]):
                prompt_len = sorted_input_ids[i+j].shape[1]
                batch_prompt_lengths.append(prompt_len)
                gen_slice = out[j, prompt_len:]
                
                eos_indices = (gen_slice == eos_id).nonzero(as_tuple=True)[0]
                if eos_indices.numel() > 0:
                    token_number = int(eos_indices[0].item())
                else:
                    token_number = int(gen_slice.shape[0])
                    
                batch_token_number += token_number
                token_numbers.append(token_number)
            
            tpf = batch_token_number/nfe/batch_size
            tps = batch_token_number/sample_time
            fps = nfe/sample_time
            tpfs.append(tpf)
            tpss.append(tps)
            fpss.append(fps)
            
            if rank == 0:
                avg_prompt_len = int(np.mean(batch_prompt_lengths))
                # Added 'prompt_len=' to the final print string
                print(f'[iter {i:4d}] nfe={nfe:4d}, prompt_len={avg_prompt_len:4d}, gen_tokens(batch)={batch_token_number:4d}, sample_time={sample_time:2.4f}, fps={fps:4.2f}({np.mean(fpss):4.2f}), tpf={tpf:2.2f}({np.mean(tpfs):4.2f}), tps={tps:4.2f}({np.mean(tpss):4.2f})')
                
                if wi==0 and i<5:
                    for j in range(min(input_ids.shape[0], 4)):
                        answer = cut_eos(out[j, sorted_input_ids[i+j].shape[1]:].unsqueeze(0), eos_id=eos_id)[0]
            total_token += token_number

        total_token = total_token

        stop = time.time()


    original_order_outputs = [None] * len(all_input_ids)
    original_order_tpfs = [None] * len(all_input_ids)
    original_order_tpss = [None] * len(all_input_ids)
    original_order_fpss = [None] * len(all_input_ids)
    original_order_token_numbers = [None] * len(all_input_ids)

    for i, original_idx in enumerate(sorted_indices):
        original_order_outputs[original_idx] = outputs[i]
        original_order_tpfs[original_idx] = tpfs[i//batch_size]
        original_order_tpss[original_idx] = tpss[i//batch_size]
        original_order_fpss[original_idx] = fpss[i//batch_size]
        original_order_token_numbers[original_idx] = token_numbers[i]

    outputs = original_order_outputs
    tpfs = original_order_tpfs
    tpss = original_order_tpss
    fpss = original_order_fpss
    token_numbers = original_order_token_numbers        

    if rank==0:
        answers = []
        for i in trange(len(outputs)):
            out = outputs[i]
            # [FIX] Apply cut_eos before decoding, otherwise trailing garbage is saved!
            prompt_len = all_input_ids[i].shape[1]
            raw_gen_tokens = out[:, prompt_len:]
            answer_tokens = cut_eos(raw_gen_tokens, eos_id=args.eos_id)[0]
            
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            answers.append(answer)
        print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/total_time}({np.mean(fpss)}), TPS: {total_token/total_time}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')
        filename = args.save_path
        with open (filename, 'w') as f:
            for i in range(len(answers)):
                answer = answers[i]
                json.dump({'answer': answer, 'generated_length': token_numbers[i], 'tpf':tpfs[i//batch_size], 'tps':tpss[i//batch_size], 'fps':fpss[i//batch_size], }, f)
                f.write('\n')

        # --- ADD TELEMETRY DUMP ---
        import dinfer.model.modeling_llada2_moe_sglang as sglang_model
        if sglang_model.GLOBAL_TELEMETRY.get("fwd_counts_decode") is not None:
            # Decode Stats (Support Sparsity)
            fwd_count_dec = sglang_model.GLOBAL_TELEMETRY["fwd_counts_decode"].cpu().item()
            expert_counts_dec = sglang_model.GLOBAL_TELEMETRY["expert_counts_decode"].cpu().numpy()
            avg_experts_per_layer_dec = (expert_counts_dec / fwd_count_dec) if fwd_count_dec > 0 else np.zeros_like(expert_counts_dec)
            apf_decode = float(np.mean(avg_experts_per_layer_dec))

            total_moe_dec_ms = sglang_model.GLOBAL_TELEMETRY["total_moe_time_ms_decode"]
            avg_moe_kernel_ms_decode = (total_moe_dec_ms / fwd_count_dec) if fwd_count_dec > 0 else 0.0

            # Prefill Stats
            fwd_count_pref = sglang_model.GLOBAL_TELEMETRY["fwd_counts_prefill"].cpu().item()
            expert_counts_pref = sglang_model.GLOBAL_TELEMETRY["expert_counts_prefill"].cpu().numpy()
            avg_experts_per_layer_pref = (expert_counts_pref / fwd_count_pref) if fwd_count_pref > 0 else np.zeros_like(expert_counts_pref)
            apf_prefill = float(np.mean(avg_experts_per_layer_pref))

            expert_histogram = sglang_model.GLOBAL_TELEMETRY["expert_histogram"].cpu()

            stats_dict = {
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "block_length": args.block_length,
                "total_time_seconds": stop - start,
                "total_forward_passes_decode": fwd_count_dec,
                "total_forward_passes_prefill": fwd_count_pref,
                "mean_fps": float(np.mean(fpss)),
                "mean_tps": float(np.mean(tpss)),
                "mean_tpf": float(np.mean(tpfs)),
                "APF_Decode (Support Sparsity)": apf_decode,
                "APF_Prefill": apf_prefill,
                "total_moe_kernel_time_s": (total_moe_dec_ms + sglang_model.GLOBAL_TELEMETRY["total_moe_time_ms_prefill"]) / 1000.0,
                "avg_moe_kernel_ms_per_fwd_decode": avg_moe_kernel_ms_decode,
                "layerwise_APF_Decode": avg_experts_per_layer_dec.tolist(),
                "layerwise_APF_Prefill": avg_experts_per_layer_pref.tolist(),
            }

            actual_save_dir = os.path.dirname(args.save_path)
            stats_file = os.path.join(actual_save_dir, "hardware_stats.json")
            with open(stats_file, 'w') as f_stats:
                json.dump(stats_dict, f_stats, indent=4)
            
            pt_file = os.path.join(actual_save_dir, "inference_telemetry.pt")
            torch.save({
                "expert_histogram": expert_histogram,
                "layerwise_APF_Decode": torch.tensor(avg_experts_per_layer_dec),
                "layerwise_APF_Prefill": torch.tensor(avg_experts_per_layer_pref),
                "metrics": stats_dict
            }, pt_file)
                
            print(f"\n[Hardware Telemetry] Saved to {stats_file} & {pt_file}")
            print(f"  -> Support Sparsity (APF_Decode): {apf_decode:.2f} unique experts/step")
            print(f"  -> Context Load (APF_Prefill): {apf_prefill:.2f} unique experts/step\n")
        # -------------------------------

        file_exists = os.path.isfile(args.speed_path) and os.path.getsize(args.speed_path) > 0
        with open(args.speed_path, 'a+') as f:
            if not file_exists:
                f.write("config,parallel_decoding,threshold,prefix_look,batch_size,block_length,total_forward,total_time,tokens_per_sample,fps_overall,tps_overall,tpf_overall,avg_padded_gen_lens,mean_fps,mean_tps,mean_tpf\n")            
            f.write(f"{args.config},{args.parallel_decoding},{args.threshold},{args.prefix_look},{args.batch_size},{args.block_length},{total_forward},{stop-start},{total_token / len(all_input_ids)},{total_forward/total_time},{total_token/total_time},{total_token/total_forward},{sum(padded_gen_lens)/total_forward},{np.mean(fpss)},{np.mean(tpss)},{np.mean(tpfs)}\n")


@dataclass
class EvalConfig:
    model_name: str = ''
    gpu: str = '0;1;2;3'
    batch_size: int = 1
    gen_len: int = 1024
    prefix_look: int = 0
    after_look: int = 0
    block_length: int = 64
    threshold: float = 0.9
    warmup_times: int = 0
    low_threshold: float = 0.3
    cont_weight: float = 0
    parallel_decoding: str = 'threshold'
    use_credit: bool = False
    cache: str = ''
    use_tp: bool = False
    save_path: str = ''
    config: int = 0
    tp_size: int = 1
    port_offset: int = 0
    all_input_ids = None
    padded_gen_lens = None
    use_cudagraph: bool = False
    use_compile: bool = True
    use_bd: bool = False
    use_shift: bool = False
    model_type: str = 'llada'
    vocab_size: int = 156896
    master_port: int = 23456
    batch_size: int = 1
    save_samples: bool = False
    speed_path: str = ''
    # ADDED:
    mask_id: int = 156900
    eos_id: int = 156892

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("dInfer_eval")
class DInferEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        device="cuda",
        mask_id=126336,
        eos_id=126081,
        vocab_size=157184,
        max_length=4096,
        batch_size=2,
        mc_num=128,
        is_check_greedy=True,
        gen_length=1024,
        block_length=1024,
        save_dir=None,
        show_speed=False,
        parallel_decoding="threshold",
        threshold: float=0.9,
        cache: str="",
        warmup_times: int=0,
        low_threshold: float=0.3,
        cont_weight: float=0,
        use_credit: bool=False,
        tp_size: int=1,
        parallel = 'dp',
        use_compile = True,
        master_port = 23456,
        use_cudagraph = True,
        gpus = '0;1;2;3',
        use_bd = False,
        prefix_look = 0,
        after_look = 0,
        use_shift = False,
        model_type = 'llada2',
        save_samples = False,
        **kwargs
    ):

        super().__init__()
        
        self.model_path = model_path
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.vocab_size = int(vocab_size)
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.parallel_decoding = parallel_decoding
        self.threshold = threshold
        self.cache = cache
        self.warmup_times = warmup_times
        self.low_threshold = low_threshold
        self.cont_weight = cont_weight
        self.use_credit = use_credit
        self.master_port = master_port
        self.tp_size = tp_size
        self.use_compile = use_compile
        self.parallel = parallel
        self.use_cudagraph = use_cudagraph
        self.gpus = gpus
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.use_bd = use_bd
        self.kwargs = kwargs
        self.use_shift = use_shift
        self.model_type = model_type
        self.save_samples = save_samples

        if self.model_type == 'llada2': 
            self.is_moe = True
        else:
            raise ValueError('model type not supported')

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerate.Accelerator()
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})  
        
            
        if parallel == 'tp':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise NotImplementedError(parallel)
        
        

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def tokenizer_name(self) -> str:
        return self.model_path
    
    def apply_chat_template(self, chat_history, **kwargs) -> str:
        if "tokenize" not in kwargs:
            kwargs["tokenize"] = False
        return self.tokenizer.apply_chat_template(chat_history, **kwargs)

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, f'rank_{self.rank}.jsonl')
            print(f"save_path: {self.save_path}")
            self.speed_path = os.path.join(self.save_dir, f'results.txt')

        

        def get_bucket_length(length):
            bucket_length = bucket_size*(length//bucket_size)
            if bucket_length not in used_buckets:
                used_buckets.append(bucket_length)
            return bucket_length

        def load_inputs(prompts, tokenizer):
            all_input_ids = []
            for id, prompt in enumerate(prompts):
                input_ids = tokenizer(prompt.args[0])['input_ids']
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                all_input_ids.append(input_ids)
            return all_input_ids

        def cal_bucket_len(gen_len, all_input_ids):
            max_prompt_length = 0
            padded_gen_lens = []

            for i in range(len(all_input_ids)):
                input_ids = all_input_ids[i]
                if input_ids.shape[1] > max_prompt_length:
                    max_prompt_length = input_ids.shape[1]
                padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
                padded_gen_lens.append(padded_length - input_ids.shape[1])
            return padded_gen_lens

        all_input_ids = load_inputs(requests, self.tokenizer)
        padded_gen_lens = cal_bucket_len(self.gen_length, all_input_ids)
    
        procs = []
        answers = []
        gpus = [int(gpu) for gpu in str(self.gpus).split(';')]
        args = {"gpu": gpus, "batch_size": self.batch_size, "model_name": self.model_path, "gen_len": self.gen_length, "block_length": self.block_length, "prefix_look": self.prefix_look, "after_look": self.after_look, "warmup_times": self.warmup_times, "low_threshold": self.low_threshold, "threshold": self.threshold, "cont_weight": self.cont_weight, "use_credit": self.use_credit, "cache": self.cache, "parallel_decoding": self.parallel_decoding, "tp_size": self.tp_size, "save_path": self.save_path, "use_cudagraph": self.use_cudagraph, "use_compile": self.use_compile,"use_bd": self.use_bd, "use_shift": self.use_shift, "model_type": self.model_type, "vocab_size": self.vocab_size, "batch_size": self.batch_size, "speed_path": self.speed_path,
                "mask_id": self.mask_id, "eos_id": self.eos_id}
        args = EvalConfig(**args)
        args.tp_size = len(gpus)
        args.master_port = self.master_port
        args.use_tp = args.tp_size > 1
        args.port_offset = gpus[0]
        args.all_input_ids = all_input_ids
        args.padded_gen_lens = padded_gen_lens
        
        if len(gpus) == 1:
            run_benchmark(1, 0, gpus[0], self.tokenizer, args)
        else:
            for i, gpu in enumerate(gpus):
                ctx = mp.get_context('spawn')
                p = ctx.Process(target=run_benchmark, args=(len(gpus), i, gpu, self.tokenizer, args))
                # p.daemon = True
                procs.append(p)
                p.start()
            for p in procs:
                p.join()
        answers = []
        with open(self.save_path, 'r') as f:
            for line in f :
                answers.append(json.loads(line)["answer"])
        if not self.save_samples is None:
            os.remove(self.save_path)
        return answers


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()