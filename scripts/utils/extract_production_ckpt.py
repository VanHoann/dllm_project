import os
import sys
import torch
from transformers import AutoConfig, AutoTokenizer
from veomni.checkpoint import ckpt_to_state_dict
from veomni.models import save_model_weights

def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_production_ckpt.py <ckpt_dir_path> <output_hf_dir> <base_merged_dir>")
        sys.exit(1)

    save_checkpoint_path = sys.argv[1]
    hf_weights_path = sys.argv[2]
    base_merged_dir = sys.argv[3]

    print(f"Loading and un-sharding distributed checkpoint from: {save_checkpoint_path}")
    
    # 1. Use the framework's native contextual un-sharder
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=save_checkpoint_path,
        output_dir=os.path.dirname(save_checkpoint_path),
        ckpt_manager="dcp",
    )

    print("Loading model assets...")
    config = AutoConfig.from_pretrained(base_merged_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_merged_dir, trust_remote_code=True)
    model_assets = [config, tokenizer]

    print(f"Saving combined weights in FusedMoE Hugging Face format to: {hf_weights_path}")
    save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
    print("Successfully extracted FusedMoE checkpoint!")

if __name__ == "__main__":
    main()