import os
import glob
import json
import torch
import itertools
from torch.utils.data import Dataset

# [TODO] PARADE Metadata JSON path
DATASET_JSON_PATH = "/mnt/data/ycevan/gen_bias/run_PARADE/PARADE_audio/audio_result_path_mapping_v2.json"
# [TODO] PARADE audio root directory
AUDIO_ROOT_DIR = "/mnt/data/ycevan/gen_bias/run_PARADE/PARADE_audio"
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
    
def load_parade_data(json_path):
    """
    Read PARADE dataset json (nested structure) and flatten into a list
    Structure: Model (onyx) -> Domain (occupation) -> Sub (programmer_typist) -> Text -> Info
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON not found at {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data_list = []
    
    # Recursive or multi-level loop parsing
    for speaker_model, domains in raw_data.items(): # e.g., onyx, nova
        for domain, subcategories in domains.items(): # e.g., occupation, status
            for subcategory, items in subcategories.items(): # e.g., programmer_typist
                for text_key, info in items.items():
                    # info contains: path, question, options, label
                    
                    # Build absolute path
                    # info['path'] example: "onyx/occupation-programmer_typist-0.mp3"
                    audio_path = os.path.join(AUDIO_ROOT_DIR, info['path'])
                    
                    item = {
                        "audio_path": audio_path,
                        "question": info['question'],
                        "options": info['options'], # list of 3 strings
                        "label": info['label'],     # ground truth
                        "speaker_model": speaker_model,
                        "domain": domain,
                        "subcategory": subcategory,
                        "text_key": text_key
                    }
                    data_list.append(item)
    
    return data_list

def generate_parade_permutations(item):
    """
    Generate all 6 permutations of PARADE options
    Input: item['options'] (e.g. ['unclear', 'programmer', 'typist'])
    Output: List of (instruction, option_map, perm_idx)
    """
    raw_options = item['options']
    
    # Generate all permutations: 3! = 6 types
    permutations = list(itertools.permutations(raw_options))
    
    results = []
    # Fixed option IDs
    option_ids = ["A", "B", "C"] 
    
    for perm_idx, perm_opts in enumerate(permutations):
        # perm_opts is a tuple, e.g. ('programmer', 'unclear', 'typist')
        
        # 1. Build option text
        # A. programmer
        # B. unclear
        # C. typist
        options_text = ""
        option_map = {}
        
        for i, opt_text in enumerate(perm_opts):
            opt_id = option_ids[i]
            options_text += f"{opt_id}. {opt_text}\n"
            option_map[opt_id] = opt_text
        
        # Remove the trailing newline
        options_text = options_text.strip()
        
        # 2. Build full Instruction
        # {question} Answer the question with one of the following options: A, B, or C:\n A. ...
        instruction = f"{item['question']} Answer the question with one of the following options: A, B, or C:\n{options_text}"
        
        results.append((instruction, option_map, perm_idx))
        
    return results