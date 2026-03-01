"""
Evaluation Script for PARADE Dataset using DeSTA 2.5 with vLLM
Features:
1. Full Permutation (6 prompts per audio) to eliminate positional bias.
2. DeSTA specific input format (<|AUDIO|> placeholder, Llama-3 template).
3. vLLM batched inference with raw audio data.
"""
import os
# [CRITICAL] Must be set before importing vllm
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import glob
import librosa
import numpy as np
import argparse
import gc
import itertools
import torch
import datetime
import random
import pandas as pd
from tqdm import tqdm
import desta.vllm # Ensure the desta package is installed
from vllm import LLM, SamplingParams
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

MODEL_ID = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
TOKENIZER_ID = "DeSTA-ntu/Llama-3.1-8B-Instruct"
AUDIO_PLACEHOLDER = "<|AUDIO|>"

# 16 data items * 6 permutations = 96 vLLM Requests
BATCH_SIZE = 16

# ================= Utility Functions =================

def prepare_desta_inputs_parade(batch_items, tokenizer, target_sr=16000):
    """
    Prepare DeSTA 2.5 vLLM inputs (PARADE version)
    """
    inputs = []
    raw_audios = []
    valid_items = []
    
    # --- Phase 1: Load audio ---
    for item in batch_items:
        path = item["audio_path"]
        try:
            # [CRITICAL] DeSTA (Whisper encoder) uses 16000 Hz
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            raw_audios.append(y)
            valid_items.append(item)
        except Exception as e:
            print(f"Error loading audio {path}: {e}")
            continue
    
    if not raw_audios:
        return []

    # --- Phase 2: Padding ---
    max_len = max([len(y) for y in raw_audios])
    padded_audios = []
    for y in raw_audios:
        if len(y) < max_len:
            y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            padded_audios.append(y_padded)
        else:
            padded_audios.append(y)

    # --- Phase 3: Generate 6 Prompt permutations ---
    for item, y_padded in zip(valid_items, padded_audios):
        
        # Get 6 permutations (from task_config)
        perm_results = generate_parade_permutations(item)
        
        for instruction, option_map, perm_idx in perm_results:
            
            # [CRITICAL] DeSTA message structure
            messages = [
                {
                    "role": "system", 
                    "content": "Focus on the audio clips and instructions."
                },
                {
                    "role": "user", 
                    # DeSTA official example format
                    "content": f"{AUDIO_PLACEHOLDER}\n\n{instruction}"
                },
            ]

            try:
                final_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Template generation failed: {e}")
                continue

            inputs.append({
                "vllm_input": {
                    "prompt": final_prompt,
                    "multi_modal_data": {"audio": (y_padded, target_sr)},
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
        exp_id = f"parade_desta_{timestamp}"
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

    # 2.2 Initialize vLLM
    try:
        llm = LLM(
            model=MODEL_ID,
            tokenizer=TOKENIZER_ID, # DeSTA specified tokenizer
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"audio": 1},
        )
        print("Getting Tokenizer from vLLM...")
        tokenizer = llm.get_tokenizer()
    except Exception as e:
        print(f"\n[Fatal Error] Initialization failed: {e}")
        return

    # Set sampling parameters (Llama-3 stop tokens)
    stop_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>"]
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=stop_tokens,
        stop_token_ids=stop_token_ids
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
            # Prepare inputs (SR=16000 for DeSTA)
            prepared_data = prepare_desta_inputs_parade(batch_items, tokenizer, target_sr=16000)
            if not prepared_data: continue
            
            vllm_inputs = [d['vllm_input'] for d in prepared_data]
            metas = [d['meta'] for d in prepared_data]

            # vLLM Generate
            outputs = llm.generate(vllm_inputs, sampling_params, use_tqdm=False)
            
            with open(output_jsonl, "a", encoding="utf-8") as f:
                for meta, output_item in zip(metas, outputs):
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
    del tokenizer
    cleanup_vllm()

    print("\nDone.")

if __name__ == "__main__":
    main()