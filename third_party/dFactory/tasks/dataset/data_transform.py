import torch
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

def process_mdm_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    block_size: int, 
    phase_block_size: int = 32,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    dynamic_noise_high: Optional[Any] = None,
    mask_token_id: int = 156900,
    source_name: Optional[str] = None,
    complementary_mask: bool = True,
) -> List[Dict[str, "torch.Tensor"]]:
    
    assert block_size >= phase_block_size and block_size % phase_block_size == 0, \
        f"Training block_size ({block_size}) must be a multiple of phase_block_size ({phase_block_size})."
    
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    examples = []
    input_ids, prompt_length = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)
    
    # --- DYNAMIC BLOCK PADDING ---
    seq_len = len(input_ids)
    pad_len = (block_size - (seq_len % block_size)) % block_size
    
    if pad_len > 0:
        pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, pad_tensor])
        
    prompt_length = min(prompt_length, len(input_ids) - 1)
    maskable_mask = torch.arange(len(input_ids)) >= prompt_length
    
    pad_mask = (input_ids == tokenizer.pad_token_id)
    maskable_mask = maskable_mask & (~pad_mask)

    segment_ids = torch.zeros_like(input_ids)
    segment_ids[pad_mask] = -1

    # -----------------------------    
    actual_response_len = maskable_mask.sum().item()
    current_high = dynamic_noise_high.value if dynamic_noise_high is not None else noise_range[1]
    current_high = max(current_high, noise_range[0])
    sigma = (torch.rand(1) * (current_high - noise_range[0]) + noise_range[0]).item()
    
    # --- View 1 Mask Generation ---
    move_indices_1 = (torch.rand(*input_ids.shape) < sigma) & maskable_mask
    noisy_input_ids_1 = torch.where(move_indices_1 | pad_mask, mask_token_id, input_ids)
    
    labels_1 = input_ids.clone()
    labels_1[:prompt_length] = -100
    labels_1[pad_mask] = -100  # Explicitly ignore pads in labels
    loss_mask_1 = (noisy_input_ids_1 == mask_token_id)
    labels_1[~loss_mask_1] = -100
    
    true_density_1 = (move_indices_1.sum().item() / actual_response_len) if actual_response_len > 0 else 0.0

    num_phase_blocks = noisy_input_ids_1.shape[0] // phase_block_size
    chunked_noisy_1 = noisy_input_ids_1.view(num_phase_blocks, phase_block_size)
    mask_counts_1 = (chunked_noisy_1 == mask_token_id).sum(dim=1, dtype=torch.long)
    block_phases_1 = mask_counts_1.unsqueeze(1).expand(-1, phase_block_size).reshape(-1)
    phase_indices_1 = torch.where(maskable_mask, block_phases_1, torch.zeros_like(block_phases_1))

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids_1,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels_1,
        "mask_density": torch.tensor(true_density_1, dtype=torch.float32),
        "segment_ids": segment_ids, 
        "phase_indices": phase_indices_1, 
    })
    
    # --- View 2 (Complementary Mask) ---
    if complementary_mask:
        move_indices_2 = (~move_indices_1) & maskable_mask
        noisy_input_ids_2 = torch.where(move_indices_2 | pad_mask, mask_token_id, input_ids)

        labels_2 = input_ids.clone()
        labels_2[:prompt_length] = -100
        labels_2[pad_mask] = -100  # Explicitly ignore pads in labels
        loss_mask_2 = (noisy_input_ids_2 == mask_token_id)
        labels_2[~loss_mask_2] = -100
        
        true_density_2 = (move_indices_2.sum().item() / actual_response_len) if actual_response_len > 0 else 0.0

        num_phase_blocks = noisy_input_ids_2.shape[0] // phase_block_size
        chunked_noisy_2 = noisy_input_ids_2.view(num_phase_blocks, phase_block_size)
        mask_counts_2 = (chunked_noisy_2 == mask_token_id).sum(dim=1, dtype=torch.long)
        block_phases_2 = mask_counts_2.unsqueeze(1).expand(-1, phase_block_size).reshape(-1)
        phase_indices_2 = torch.where(maskable_mask, block_phases_2, torch.zeros_like(block_phases_2))

        examples.append({
            "input_ids": input_ids,
            "noisy_input_ids": noisy_input_ids_2,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels_2,
            "mask_density": torch.tensor(true_density_2, dtype=torch.float32), 
            "segment_ids": segment_ids,
            "phase_indices": phase_indices_2,
        })
        
    return examples


def process_mdm_tokenized_example(
    example: Dict[str, List[int]],
    max_seq_len: int, 
    block_size: int, 
    phase_block_size: int = 32,
    text_keys: Union[str, List[str]] = "input_ids",
    noise_range: Tuple[float, float] = (0.1, 0.9), 
    dynamic_noise_high: Optional[Any] = None,
    mask_token_id: int = 156900,                   
    pad_token_id: int = 156892,
    source_name: Optional[str] = None,
    complementary_mask: bool = False,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        input_ids = example[text_keys]
    else:
        for text_key in text_keys:
            if text_key in example:
                input_ids = example[text_key]
                break
    
    prompt_length = example.get('prompt_lengths', 0)
    input_ids = torch.tensor(input_ids)
    
    # --- EXTRACT SEGMENT IDS ---
    if "segment_ids" in example:
        segment_ids = torch.tensor(example["segment_ids"])
    else:
        segment_ids = torch.zeros_like(input_ids)
    
    # Dynamic padding just in case (though Arrow should already be 2048)
    seq_len = len(input_ids)
    pad_len = (block_size - (seq_len % block_size)) % block_size
    
    if pad_len > 0:
        input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        segment_ids = torch.cat([segment_ids, torch.full((pad_len,), -1, dtype=torch.long)])
        
    prompt_length = min(prompt_length, len(input_ids) - 1)
    maskable_mask = torch.arange(len(input_ids)) >= prompt_length
    
    # Padding defined by segment_ids == -1 or input_ids == pad_token_id
    pad_mask = (segment_ids == -1) | (input_ids == pad_token_id)
    maskable_mask = maskable_mask & (~pad_mask)
    
    actual_response_len = maskable_mask.sum().item() 
    
    current_high = dynamic_noise_high.value if dynamic_noise_high is not None else noise_range[1]
    current_high = max(current_high, noise_range[0])
    sigma = (torch.rand(1) * (current_high - noise_range[0]) + noise_range[0]).item()
    
    # --- View 1 Mask Generation ---
    move_indices_1 = (torch.rand(*input_ids.shape) < sigma) & maskable_mask
    noisy_input_ids_1 = torch.where(move_indices_1 | pad_mask, mask_token_id, input_ids)
    
    labels_1 = input_ids.clone()
    labels_1[:prompt_length] = -100
    labels_1[pad_mask] = -100
    labels_1[~(noisy_input_ids_1 == mask_token_id)] = -100
    
    true_density_1 = (move_indices_1.sum().item() / actual_response_len) if actual_response_len > 0 else 0.0 
    
    num_phase_blocks = noisy_input_ids_1.shape[0] // phase_block_size
    chunked_noisy_1 = noisy_input_ids_1.view(num_phase_blocks, phase_block_size)
    mask_counts_1 = (chunked_noisy_1 == mask_token_id).sum(dim=1, dtype=torch.long)
    block_phases_1 = mask_counts_1.unsqueeze(1).expand(-1, phase_block_size).reshape(-1)
    phase_indices_1 = torch.where(maskable_mask, block_phases_1, torch.zeros_like(block_phases_1))

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids_1,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels_1,
        "mask_density": torch.tensor(true_density_1, dtype=torch.float32), 
        "segment_ids": segment_ids, 
        "phase_indices": phase_indices_1,
    })
    
    # --- View 2 (Complementary Mask) ---
    if complementary_mask:
        move_indices_2 = (~move_indices_1) & maskable_mask
        noisy_input_ids_2 = torch.where(move_indices_2 | pad_mask, mask_token_id, input_ids)
        
        labels_2 = input_ids.clone()
        labels_2[:prompt_length] = -100
        labels_2[pad_mask] = -100 
        labels_2[~(noisy_input_ids_2 == mask_token_id)] = -100
        
        true_density_2 = (move_indices_2.sum().item() / actual_response_len) if actual_response_len > 0 else 0.0 
        
        num_phase_blocks = noisy_input_ids_2.shape[0] // phase_block_size
        chunked_noisy_2 = noisy_input_ids_2.view(num_phase_blocks, phase_block_size)
        mask_counts_2 = (chunked_noisy_2 == mask_token_id).sum(dim=1, dtype=torch.long)
        block_phases_2 = mask_counts_2.unsqueeze(1).expand(-1, phase_block_size).reshape(-1)
        phase_indices_2 = torch.where(maskable_mask, block_phases_2, torch.zeros_like(block_phases_2))

        examples.append({
            "input_ids": input_ids,
            "noisy_input_ids": noisy_input_ids_2,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels_2,
            "mask_density": torch.tensor(true_density_2, dtype=torch.float32), 
            "segment_ids": segment_ids,
            "phase_indices": phase_indices_2,
        })
        
    return examples



def sft_noise_transition(x_0, noise_range, maskable_mask, mask_token_id):
    t_tensor = torch.rand(1) * (noise_range[1] - noise_range[0]) + noise_range[0]
    sigma = t_tensor.item()
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t


def apply_chat_template_mdm(messages, tokenizer, max_length):
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_ids_unpadded)

    # REMOVED padding="max_length"
    tokenized_input = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True, 
        max_length=max_length, 
        add_special_tokens=False
    ).input_ids.squeeze(0)

    return tokenized_input, prompt_length