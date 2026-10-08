import os
import torch
import json

def main():
    scratch_root = "/ictstr01/scratch/users/van.trinh"
    merged_dir = f"{scratch_root}/checkpoints/Ling-mini-2.0-to-LLaDA2.0-mini-merged"
    
    print(f"Scanning merged directory: {merged_dir}")
    
    # 1. Process safetensors if they exist
    safetensors_files = [f for f in os.listdir(merged_dir) if f.endswith(".safetensors")]
    if len(safetensors_files) > 0:
        from safetensors.torch import load_file, save_file
        print(f"Found {len(safetensors_files)} safetensors file(s). Performing surgery...")
        for f in safetensors_files:
            path = os.path.join(merged_dir, f)
            state_dict = load_file(path)
            modified = False
            
            if "model.word_embeddings.weight" in state_dict:
                # Copy original role_end (156895) to new mask position (156900)
                state_dict["model.word_embeddings.weight"][156900] = state_dict["model.word_embeddings.weight"][156895].clone()
                print(f"-> Successfully cloned embedding weights in {f}")
                modified = True
                
            if "lm_head.weight" in state_dict:
                state_dict["lm_head.weight"][156900] = state_dict["lm_head.weight"][156895].clone()
                print(f"-> Successfully cloned lm_head weights in {f}")
                modified = True
                
            if modified:
                save_file(state_dict, path)

    # 2. Process PyTorch .bin/.pt files if they exist
    bin_files = [f for f in os.listdir(merged_dir) if f.endswith(".bin") or f.endswith(".pt")]
    if len(bin_files) > 0:
        print(f"Found {len(bin_files)} PyTorch binary file(s). Performing surgery...")
        for f in bin_files:
            path = os.path.join(merged_dir, f)
            state_dict = torch.load(path, map_location="cpu")
            modified = False
            
            if "model.word_embeddings.weight" in state_dict:
                state_dict["model.word_embeddings.weight"][156900] = state_dict["model.word_embeddings.weight"][156895].clone()
                print(f"-> Successfully cloned embedding weights in {f}")
                modified = True
                
            if "lm_head.weight" in state_dict:
                state_dict["lm_head.weight"][156900] = state_dict["lm_head.weight"][156895].clone()
                print(f"-> Successfully cloned lm_head weights in {f}")
                modified = True
                
            if modified:
                torch.save(state_dict, path)
                
    print("Weight surgery complete!")

if __name__ == "__main__":
    main()