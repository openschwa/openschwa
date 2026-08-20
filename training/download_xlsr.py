"""Download and convert XLS-R 300M to safetensors format."""

from __future__ import annotations

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
from safetensors.torch import save_file

def download_and_convert() -> None:
    print("Downloading config and weights...")
    out_dir = Path("/home/boxcanyon/OpenSchwa/openschwa/.models/wav2vec2-xls-r-300m")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Download config
    hf_hub_download("facebook/wav2vec2-xls-r-300m", "config.json", local_dir=str(out_dir))
    try:
        hf_hub_download("facebook/wav2vec2-xls-r-300m", "preprocessor_config.json", local_dir=str(out_dir))
    except Exception:
        pass
        
    # Check if safetensors exists
    try:
        hf_hub_download("facebook/wav2vec2-xls-r-300m", "model.safetensors", local_dir=str(out_dir))
        print("Downloaded model.safetensors directly!")
    except Exception:
        print("Downloading pytorch_model.bin and converting to safetensors...")
        bin_file = hf_hub_download("facebook/wav2vec2-xls-r-300m", "pytorch_model.bin")
        state_dict = torch.load(bin_file, map_location="cpu", weights_only=False)
        save_file(state_dict, str(out_dir / "model.safetensors"))
        print("Converted and saved model.safetensors successfully!")
    
    print("XLS-R 300M ready in", out_dir)

if __name__ == "__main__":
    download_and_convert()
