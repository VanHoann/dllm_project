import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from pathlib import Path

def main():
    scratch_root = "/ictstr01/scratch/users/van.trinh"
    model_path = f"{scratch_root}/checkpoints/Ling-mini-2.0-to-LLaDA2.0-mini-merged"
    output_dir = f"{scratch_root}/data/smollm_cpt_arrow_old"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 156892
    max_length = 2048

    print("Streaming mixed pretraining datasets...")
    fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True).take(7_000_000) 
    cosmopedia = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True).take(3_500_000)
    python_edu = load_dataset("Avelina/python-edu-cleaned", split="train", streaming=True).take(2_000_000)

    # Combined generator function to feed Arrow directly
    def global_tokenization_generator():
        datasets_mix = [
            (fineweb, "FineWeb-Edu"),
            (cosmopedia, "Cosmopedia-v2"),
            (python_edu, "Python-Edu")
        ]
        
        for dataset, source_name in datasets_mix:
            print(f"Tokenizing and segmenting streaming source: {source_name}...")
            token_buffer = []
            segment_buffer = []
            current_doc_id = 0
            
            for row in dataset:
                text = row.get("text", "")
                if not text or not text.strip():
                    continue
                    
                tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
                doc_tokens = tokens + [eos_token_id]
                doc_segments = [current_doc_id] * len(doc_tokens)
                
                token_buffer.extend(doc_tokens)
                segment_buffer.extend(doc_segments)
                current_doc_id += 1
                
                while len(token_buffer) >= max_length:
                    chunk_tokens = token_buffer[:max_length]
                    chunk_segments = segment_buffer[:max_length]
                    
                    unique_segments = sorted(list(set(chunk_segments)))
                    segment_mapping = {orig: new for new, orig in enumerate(unique_segments)}
                    normalized_segments = [segment_mapping[sid] for sid in chunk_segments]
                    
                    yield {
                        "input_ids": chunk_tokens,
                        "segment_ids": normalized_segments,
                        "prompt_lengths": 0
                    }
                    
                    token_buffer = token_buffer[max_length:]
                    segment_buffer = segment_buffer[max_length:]
                        
            if len(token_buffer) > 0:
                pad_len = max_length - len(token_buffer)
                chunk_tokens = token_buffer + [eos_token_id] * pad_len
                chunk_segments = segment_buffer + [-1] * pad_len
                
                unique_segments = sorted(list(set([s for s in chunk_segments if s != -1])))
                segment_mapping = {orig: new for new, orig in enumerate(unique_segments)}
                segment_mapping[-1] = -1
                normalized_segments = [segment_mapping[sid] for sid in chunk_segments]
                
                yield {
                    "input_ids": chunk_tokens,
                    "segment_ids": normalized_segments,
                    "prompt_lengths": 0
                }

    print("Building HF Dataset via disk-backed Arrow streaming...")
    # This automatically writes data directly to files in HF_HOME cache instead of RAM
    hf_dataset = Dataset.from_generator(global_tokenization_generator)
    
    print("Executing global deterministic shuffle...")
    hf_dataset = hf_dataset.shuffle(seed=42)
    
    total_rows = len(hf_dataset)
    print(f"Grand Total Packed CPT Sequences generated: {total_rows}")
    
    eval_split = int(total_rows * 0.01)
    train_rows = total_rows - eval_split
    warmup_end = int(train_rows * 0.20)
    stable_end = int(train_rows * 0.80)
    
    splits = {
        "warmup": hf_dataset.select(range(0, warmup_end)),
        "stable": hf_dataset.select(range(warmup_end, stable_end)),
        "decay": hf_dataset.select(range(stable_end, train_rows)),
        "eval": hf_dataset.select(range(train_rows, total_rows))
    }
    
    for phase_name, ds in splits.items():
        phase_dir = os.path.join(output_dir, phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        print(f"Saving pre-tokenized {phase_name} split ({len(ds)} rows) to: {phase_dir}")
        ds.save_to_disk(phase_dir)

if __name__ == "__main__":
    main()

# output
# Grand Total Packed CPT Sequences generated: 5204301
# Saving pre-tokenized warmup split (1030451 rows) to: /ictstr01/scratch/users/van.trinh/data/smollm_cpt_arrow_old/warmup
# Saving pre-tokenized stable split (3091355 rows) to: /ictstr01/scratch/users/van.trinh/data/smollm_cpt_arrow_old/stable
# Saving pre-tokenized decay split (1030452 rows) to: /ictstr01/scratch/users/van.trinh/data/smollm_cpt_arrow_old/decay
# Saving pre-tokenized eval split (52043 rows) to: /ictstr01/scratch/users/van.trinh/data/smollm_cpt_arrow_old/eval
# merge train splits to manege later easier

# ==================================================
# CURRENT LOCAL SPLIT SIZES
# ==================================================
# Warmup split: 1,030,451 sequences (~2,110,363,648 tokens)
# Stable split: 3,091,355 sequences (~6,331,095,040 tokens)
# Decay split:  1,030,452 sequences (~2,110,365,696 tokens)
# Eval split:   52,043 sequences (~106,584,064 tokens)

# Merging warmup, stable, and decay into a unified training set...

# ==================================================
# CONSOLIDATED DATASET SUMMARY
# ==================================================
# Final TRAIN split: 5,152,258 sequences (~10,551,824,384 tokens)
# Final EVAL split:  52,043 sequences (~106,584,064 tokens)
# Grand Total:       5,204,301 sequences (~10,658,408,448 tokens)
# ==================================================
