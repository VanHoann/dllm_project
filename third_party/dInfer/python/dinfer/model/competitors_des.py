import torch

def compute_des_vote_mask(
    router_logits: torch.Tensor, 
    top_k: int, 
    m_core: int, 
    bsz: int, 
    seq_len: int, 
    mode: str = "per_sequence"
) -> torch.Tensor:
    """
    Computes the DES-Vote expert mask.
    router_logits: [num_tokens, num_experts] (where num_tokens = bsz * seq_len)
    """
    num_experts = router_logits.shape[-1]
    num_tokens = router_logits.shape[0]
    
    # 1. Mask out weights falling outside each token's local Top-K
    _, local_topk_idx = torch.topk(router_logits, top_k, dim=-1)
    local_mask = torch.zeros_like(router_logits, dtype=torch.bool).scatter_(-1, local_topk_idx, True)
    
    # Use Sigmoid weights for voting (matching LLaDA's base routing logic)
    weights = torch.sigmoid(router_logits.float())
    masked_weights = weights * local_mask.float()
    
    if mode == "per_sequence":
        # Intra-Sequence Sharing: Vote across the tokens of a single sequence
        # Failsafe for varying lengths: reshape dynamically
        actual_bsz = num_tokens // seq_len
        masked_weights = masked_weights.view(actual_bsz, seq_len, num_experts)
        
        votes = masked_weights.sum(dim=1)  # [actual_bsz, num_experts]
        
        # Select Top M_core experts per sequence
        _, coreset_idx = torch.topk(votes, m_core, dim=-1)  # [actual_bsz, m_core]
        seq_mask = torch.zeros_like(votes, dtype=torch.bool).scatter_(-1, coreset_idx, True)
        
        # Expand back to tokens and flatten
        token_mask = seq_mask.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, num_experts)
        
    elif mode == "whole_batch":
        # Global Batch Sharing: Vote across ALL tokens in the batch
        votes = masked_weights.sum(dim=0)  # [num_experts]
        
        # Select global Top M_core experts
        _, coreset_idx = torch.topk(votes, m_core, dim=-1)  # [m_core]
        global_mask = torch.zeros_like(votes, dtype=torch.bool).scatter_(-1, coreset_idx, True)
        
        # Expand to all tokens in the batch
        token_mask = global_mask.unsqueeze(0).expand(num_tokens, -1)
        
    else:
        raise ValueError(f"Unknown DES-Vote mode: {mode}")

    return token_mask