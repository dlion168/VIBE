"""
Evaluation Script for PARADE Dataset using Phi-4-multimodal with vLLM
Features:
1. Full Permutation (6 prompts per audio) to eliminate positional bias.
2. Phi-4 specific LoRA loading for speech capability.
3. Phi-4 prompt template (<|audio_1|>).
"""
import os
# [CRITICAL] Must be set before importing vllm
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import json
import glob
import librosa
import numpy as np
import argparse
import torch
import gc
import datetime
import itertools
import pandas as pd
from tqdm import tqdm

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel

# [CRITICAL] Import shared configuration and PARADE-specific functions
from task_config import (
    get_processed_files, 
    ensure_dir, 
    load_parade_data, 
    generate_parade_permutations, 
    DATASET_JSON_PATH, 
    AUDIO_ROOT_DIR
)

# ================= Configuration =================
RESULTS_ROOT = "./results"

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

# 16 data items * 6 permutations = 96 vLLM Requests
BATCH_SIZE = 16

# ================= Utility Functions =================

def prepare_phi4_inputs_parade(batch_items, target_sr=16000):
    """
    Prepare Phi-4 vLLM inputs (PARADE version)
    Features:
    1. Audio Loading (16k)
    2. Phi-4 Prompt Template (<|user|><|audio_1|>{instruction}<|end|><|assistant|>)
    3. Full Permutation
    """
    inputs = []
    
    # 1. Load audio (Cache audio to avoid reloading for permutations)
    raw_audios = {} # path -> (y, sr)
    
    for item in batch_items:
        path = item["audio_path"]
        if path not in raw_audios:
            try:
                # Phi-4 uses 16000 Hz
                y, sr = librosa.load(path, sr=target_sr, mono=True)
                raw_audios[path] = (y, sr)
            except Exception as e:
                print(f"Error loading audio {path}: {e}")
                continue

    # 2. Generate prompts for 6 permutations
    for item in batch_items:
        path = item["audio_path"]
        if path not in raw_audios: continue
        
        audio_data = raw_audios[path] # (y, sr)
        
        # Get 6 permutations
        perm_results = generate_parade_permutations(item)
        
        for instruction, option_map, perm_idx in perm_results:
            
            # [CRITICAL] Phi-4 Prompt format
            # Uses <|audio_1|> tag
            prompt_text = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"

            inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {
                    "audio": audio_data 
                },
                "meta": {
                    "path": item["audio_path"],
                    "option_map": option_map, # Key: record A/B/C to text mapping
                    "instruction": instruction,
                    "label": item["label"],
                    "speaker_model": item["speaker_model"],
                    "domain": item["domain"],
                    "perm_idx": perm_idx
                }
            })
        
    return inputs

def cleanup_vllm():
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU Memory released.")

# ================= Main Program =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of test samples")
    parser.add_argument("-e", "--exp_id", type=str, default=None, help="Experiment ID")
    return parser.parse_args()

def main():
    args = parse_args()

    # Use the imported path
    print(f"Loading PARADE dataset: {DATASET_JSON_PATH}")
    
    try:
        # Use the imported loader
        dataset = load_parade_data(DATASET_JSON_PATH)
    except Exception as e:
        print(f"Loading failed: {e}")
        return
    
    if not dataset:
        print(f"Error: No data found.")
        return
    
    print(f"Loaded {len(dataset)} raw data items (expected to produce {len(dataset)*6} inferences)。")

    if args.limit:
        print(f"[Test mode] Using only the first {args.limit} items")
        dataset = dataset[:args.limit]

    # Sort to keep batch stable
    dataset.sort(key=lambda x: x['audio_path'])

    if args.exp_id:
        exp_id = args.exp_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"parade_phi4_{timestamp}"
        if args.limit: exp_id += "_test"

    # PARADE results are not separated by dimension folder, placed directly under exp_id
    exp_dir = os.path.join(RESULTS_ROOT, exp_id)
    ensure_dir(exp_dir)
    print(f"Results output directory: {exp_dir}")

    # Run model
    model_short_name = MODEL_ID.split("/")[-1]
    print(f"\n{'='*40}")
    print(f"Preparing to run model: {model_short_name}")
    print(f"{'='*40}")

    # [CRITICAL] Download model Snapshot to get speech-lora path
    print("Checking/downloading model Snapshot...")
    model_path = snapshot_download(MODEL_ID)
    speech_lora_path = os.path.join(model_path, "speech-lora")
    print(f"Speech LoRA path: {speech_lora_path}")

    # 2.2 Initialize vLLM (enable LoRA)
    try:
        llm = LLM(
            model=model_path, 
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"audio": 1},
            enable_lora=True,    # Enable LoRA
            max_lora_rank=320,   # Phi-4 requirement
            enforce_eager=True,
        )
    except Exception as e:
        print(f"\n[Fatal Error] vLLM Initialization failed: {e}")
        return

    # [CRITICAL] Create Speech LoRA Request
    speech_lora_req = LoRARequest("speech", 1, speech_lora_path)

    # Set sampling parameters
    stop_tokens = ["<|end|>", "<|endoftext|>"]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=stop_tokens,
    )

    output_jsonl = os.path.join(exp_dir, f"PARADE_{model_short_name}_generations.jsonl")
    
    processed_files = get_processed_files(output_jsonl)
    items_to_run = [d for d in dataset if os.path.basename(d["audio_path"]) not in processed_files]
    
    if not items_to_run:
        print(f"All completed, skipping.")
        return
        
    print(f"\nRemaining {len(items_to_run)} raw data items")

    chunk_size = BATCH_SIZE
    
    for i in tqdm(range(0, len(items_to_run), chunk_size), desc=f"{model_short_name}"):
        batch_items = items_to_run[i : i + chunk_size]
        
        try:
            # Prepare inputs (SR=16000 for Phi-4)
            inputs_list = prepare_phi4_inputs_parade(batch_items, target_sr=16000)
            if not inputs_list: continue

            # vLLM Generate (pass lora_request)
            outputs = llm.generate(
                inputs_list, 
                sampling_params, 
                use_tqdm=False,
                lora_request=speech_lora_req 
            )
            
            with open(output_jsonl, "a", encoding="utf-8") as f:
                for input_item, output_item in zip(inputs_list, outputs):
                    meta = input_item["meta"]
                    generated_text = output_item.outputs[0].text.strip()
                    
                    record = {
                        "audio_file": os.path.basename(meta["path"]), 
                        "full_path": meta["path"],
                        "model": model_short_name,
                        "speaker_model": meta["speaker_model"], 
                        "domain": meta["domain"],
                        "prompt": meta["instruction"],
                        "response": generated_text,
                        "option_map": meta["option_map"],
                        "label": meta["label"],
                        "perm_idx": meta["perm_idx"],
                        "task": "PARADE"
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        except Exception as e:
            print(f"[Error] Batch failed: {e}")
            torch.cuda.empty_cache()
            continue

    del llm
    cleanup_vllm()

    print("\nDone.")

if __name__ == "__main__":
    main()