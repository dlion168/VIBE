import os
import glob
import json
import torch
from torch.utils.data import Dataset
# ================= Shared Dataset Class =================
class AudioDataset(Dataset):
    def __init__(self, audio_files, prompt_text):
        self.audio_files = audio_files
        self.prompt_text = prompt_text

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return {
            "audio_path": self.audio_files[idx],
            "prompt": self.prompt_text
        }

# ================= Utility Functions =================
def get_processed_files(jsonl_path):
    """Read the list of completed files, used for resuming"""
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data.get("audio_file"))
            except json.JSONDecodeError:
                continue
    return processed

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def get_audio_id_from_path(path, dimension):
    """
    [New] Unified logic for generating audio_id
    Ensures the ID format is identical when checking for resume and when writing files
    """
    filename = os.path.basename(path)
    if dimension == "accent":
        # Accent format: ParentGrand/Parent/filename.wav (e.g., ABA/wav/arctic_001.wav)
        parent_dir = os.path.basename(os.path.dirname(path))
        parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return os.path.join(parent_parent_dir, parent_dir, filename)
    else:
        # Gender format: filename.wav
        return filename