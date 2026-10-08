import torch

def compute_dmoe_mask(rank_scores: torch.Tensor, bsz: int, seq_len: int, top_k: int, p: float = 0.6, guard_size: int = 16, mode: str = "per_sequence") -> torch.Tensor:
    """
    Computes the dMoE coreset mask based on Top-P cumulative probability.
    mode="per_sequence": Original dMoE (aggregates within each sequence's block).
    mode="whole_batch": "Stretched" dMoE (aggregates across all sequences in the batch).
    """
    # 1. Reshape to block level: [bsz, seq_len, num_experts]
    block_scores = rank_scores.view(bsz, seq_len, -1)
    
    # 2. Local Top-K Filtering
    local_scores, local_idx = torch.topk(block_scores, k=top_k, dim=-1)
    masked_scores = torch.zeros_like(block_scores)
    masked_scores.scatter_(-1, local_idx, local_scores)
    
    # 3. Aggregate Votes
    if mode == "per_sequence":
        # Original dMoE: Sum across the sequence length [bsz, num_experts]
        votes = masked_scores.sum(dim=1)
    elif mode == "whole_batch":
        # Stretched dMoE: Sum across the entire batch AND sequence length [1, num_experts]
        # FIX: Avoid keepdim=True which creates a [1, 1, num_experts] 3D tensor
        votes = masked_scores.sum(dim=(0, 1)).unsqueeze(0)
    else:
        raise ValueError(f"Unknown dMoE mode: {mode}")
    
    # 4. Normalize
    votes = votes / (votes.sum(dim=-1, keepdim=True) + 1e-20)
    
    # 5. Dynamic coreset selection by Top-P
    sorted_votes, sorted_idx = torch.sort(votes, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_votes, dim=-1)
    
    sorted_keep = cum_probs <= p
    sorted_keep[..., 0] = True  # Always keep the top 1 expert
    
    exceed = cum_probs > p
    exceed_prev = torch.cat([torch.zeros_like(exceed[..., :1]), exceed[..., :-1]], dim=-1)
    first_exceed = exceed & (~exceed_prev)
    sorted_keep = sorted_keep | first_exceed
    
    coreset_mask = torch.zeros_like(votes, dtype=torch.bool)
    
    # FIX: Use dim -1 instead of 1 to ensure it always scatters across the expert dimension
    coreset_mask.scatter_(-1, sorted_idx, sorted_keep)
    
    # 6. Safeguard: ensure at least guard_size experts are kept
    num_selected = coreset_mask.sum(dim=-1)
    need_fix = num_selected < guard_size
    if need_fix.any():
        fallback_idx = torch.topk(votes, k=guard_size, dim=-1).indices
        fallback_mask = torch.zeros_like(coreset_mask, dtype=torch.bool)
        # FIX: Use dim -1 here too
        fallback_mask.scatter_(-1, fallback_idx, True)
        coreset_mask = torch.where(need_fix.unsqueeze(-1), fallback_mask, coreset_mask)
        
    # 7. Broadcast the mask back to individual tokens
    if mode == "per_sequence":
        token_coreset_mask = coreset_mask.unsqueeze(1).expand(-1, seq_len, -1).reshape(bsz * seq_len, -1)
    elif mode == "whole_batch":
        # FIX: Even if expanding [1, num_experts] to [bsz * seq_len, num_experts], explicit reshaping prevents edge-cases
        token_coreset_mask = coreset_mask.view(1, -1).expand(bsz * seq_len, -1)
    
    return token_coreset_mask