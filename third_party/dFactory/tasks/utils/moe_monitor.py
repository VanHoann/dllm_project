import os
import torch
import torch.distributed as dist
from veomni.distributed.parallel_state import get_parallel_state

def unwrap_module(module):
    """
    Recursively unwraps FSDP wrappers to access the underlying PyTorch module attributes.
    """
    while hasattr(module, "_fsdp_wrapped_module"):
        module = module._fsdp_wrapped_module
    return module

class MoEMonitor:
    def __init__(self, model, num_experts=256, num_layers=20, first_k_dense_replace=1, phase_bins=33, log_dir="logs/moe_stats"):
        self.model = model
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.log_dir = log_dir
        self.num_bins = phase_bins
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset_accumulators()

    def reset_accumulators(self):
        num_moe_layers = self.num_layers - self.first_k_dense_replace
        
        # Accumulate activation frequency locally: [MoE Layers, Experts, Mask Bins]
        self.accumulated_activations = torch.zeros(
            (num_moe_layers, self.num_experts, self.num_bins), 
            dtype=torch.long, 
            device="cuda"
        )
        # Accumulate representative gradient norms locally: [MoE Layers, Experts]
        self.accumulated_grad_norms = torch.zeros(
            (num_moe_layers, self.num_experts), 
            dtype=torch.float32, 
            device="cuda"
        )
        self.accumulated_steps = 0

    @torch.no_grad()
    def update_activations(self, phase_indices: torch.Tensor, valid_mask: torch.Tensor, router_logits_tuple: tuple):
        """
        Calculates exact token-level phase distributions.
        phase_indices: [batch_size, seq_len] containing 0 to 32.
        valid_mask: [batch_size, seq_len] boolean mask to filter out padding.
        """
        if router_logits_tuple is None:
            return
        
        for idx, layer_outputs in enumerate(router_logits_tuple):
            if layer_outputs is None:
                continue
            
            # layer_outputs is a tuple: (router_logits, topk_idx)
            topk_idx = layer_outputs[1]  # shape: [bsz, seq_len, top_k]
            
            # 1. Extract only valid tokens (ignores sequence padding)
            v_topk = topk_idx[valid_mask].flatten() # [N_valid * top_k]
            
            # 2. Extract corresponding phases and duplicate them to match top_k routing
            v_phase = phase_indices[valid_mask].unsqueeze(1).expand(-1, topk_idx.size(-1)).flatten().to(v_topk.device)
            
            # 3. Flatten 2D coords into 1D for ultra-fast bincounting
            # flat_idx maps to a unique index for every (Phase, Expert) combination
            flat_idx = v_phase * self.num_experts + v_topk
            
            # 4. Count occurrences and reshape back to [Phase_Bins, Experts]
            counts_1d = torch.bincount(flat_idx, minlength=self.num_bins * self.num_experts)
            counts_2d = counts_1d.view(self.num_bins, self.num_experts)
            
            # 5. Add to accumulated tensor. 
            # Transpose counts_2d from [Bins, Experts] to match [Experts, Bins] expected by accumulated_activations
            self.accumulated_activations[idx] += counts_2d.t()

    @torch.no_grad()
    def update_gradients(self):
        """
        Extracts local sharded gradients, gathers them across Expert Parallelism, and averages.
        Called after loss.backward() and before optimizer.step().
        """
        self.accumulated_steps += 1
        num_moe_layers = self.num_layers - self.first_k_dense_replace
        
        # Unwrap the top level model wrapper (LLaDA2MoeModelLM)
        raw_model = unwrap_module(self.model)
        
        # Access the sub-model (LLaDA2MoeModel) containing the layers
        base_model = unwrap_module(raw_model.model) if hasattr(raw_model, "model") else raw_model
        
        for idx in range(num_moe_layers):
            layer_id = idx + self.first_k_dense_replace
            
            # Unwrap nested layers recursively
            layer = unwrap_module(base_model.layers[layer_id])
            mlp = unwrap_module(layer.mlp)
            experts = unwrap_module(mlp.experts)
            
            # Retrieve the gate_proj parameter (representative of the FFN updates)
            param = experts.gate_proj
            
            if param.grad is not None:
                grad = param.grad
                # Gracefully extract physical tensor if wrapped in DTensor
                local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
                
                # Compute L2 norm of gradients per expert [local_num_experts]
                local_norms = local_grad.float().norm(p=2, dim=(1, 2))
                
                # Safe EP Group lookup to handle ep_size=1 configurations gracefully
                ep_group = None
                if dist.is_initialized():
                    try:
                        ep_group = get_parallel_state().ep_group
                    except (TypeError, KeyError, AttributeError):
                        ep_group = None

                # Gather sharded expert stats across EP ranks if EP is active
                if ep_group is not None and dist.get_world_size(ep_group) > 1:
                    full_norms = [torch.zeros_like(local_norms) for _ in range(dist.get_world_size(ep_group))]
                    dist.all_gather(full_norms, local_norms, group=ep_group)
                    global_norms = torch.cat(full_norms, dim=0)
                else:
                    global_norms = local_norms
                
                self.accumulated_grad_norms[idx] += global_norms

    def save(self, global_step: int):
        """
        Aggregates gathered statistics across Data Parallel ranks and saves on Rank 0.
        """
        if self.accumulated_steps > 0:
            self.accumulated_grad_norms /= self.accumulated_steps
            
        dp_group = None
        if dist.is_initialized():
            try:
                dp_group = get_parallel_state().fsdp_group
            except (TypeError, KeyError, AttributeError):
                dp_group = None

        if dp_group is not None and dist.get_world_size(dp_group) > 1:
            dist.all_reduce(self.accumulated_activations, op=dist.ReduceOp.SUM, group=dp_group)
            dist.all_reduce(self.accumulated_grad_norms, op=dist.ReduceOp.SUM, group=dp_group)
            self.accumulated_grad_norms /= dist.get_world_size(dp_group)

        expert_biases = []
        raw_model = unwrap_module(self.model)
        base_model = unwrap_module(raw_model.model) if hasattr(raw_model, "model") else raw_model
        
        num_moe_layers = self.num_layers - self.first_k_dense_replace
        for idx in range(num_moe_layers):
            layer_id = idx + self.first_k_dense_replace
            layer = unwrap_module(base_model.layers[layer_id])
            mlp = unwrap_module(layer.mlp)
            
            # Extract from the gate if it's a sparse MoE block
            if hasattr(mlp, "gate"):
                gate = unwrap_module(mlp.gate)
                if hasattr(gate, "expert_bias"):
                    expert_biases.append(gate.expert_bias.detach().cpu().clone())
        
        if len(expert_biases) > 0:
            expert_biases_tensor = torch.stack(expert_biases) # [Layers, Phase_Bins, Experts]
        else:
            expert_biases_tensor = None
            
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_path = os.path.join(self.log_dir, f"moe_stats_step_{global_step}.pt")
            torch.save({
                "global_step": global_step,
                "activation_counts": self.accumulated_activations.cpu(),
                "grad_norms": self.accumulated_grad_norms.cpu(),
                "expert_biases": expert_biases_tensor, # Logged for offline extraction and plotting
            }, save_path)
            
        self.reset_accumulators()