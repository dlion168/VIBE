"""
Evaluation Script for PARADE Dataset using Gemma-3n with vLLM
Features:
1. Full Permutation (6 prompts per audio) to eliminate positional bias.
2. Gemma-3n specific input format (SR=24000, structured content).
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
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer

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

MODELS_TO_RUN = [
    "google/gemma-3n-E2B-it",
    "google/gemma-3n-E4B-it"
]

# 16 data items * 6 permutations = 96 vLLM Requests
BATCH_SIZE = 16

# ================= Utility Functions =================

def prepare_gemma_inputs_parade(batch_items, tokenizer, target_sr=24000):
    """
    Prepare Gemma-3n vLLM inputs (PARADE version)
    Features:
    1. Audio Padding (align to longest in batch)
    2. Gemma Chat Template structure
    """
    inputs = []
    raw_audios = []
    valid_items = []
    
    # --- Phase 1: Load audio ---
    for item in batch_items:
        path = item["audio_path"]
        try:
            # [CRITICAL] Gemma-3n recommends using 24000 Hz
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
            # Zero-pad to maximum length
            y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            padded_audios.append(y_padded)
        else:
            padded_audios.append(y)

    # --- Phase 3: Generate 6 Prompt permutations ---
    for item, y_padded in zip(valid_items, padded_audios):
        
        # Get 6 permutations
        perm_results = generate_parade_permutations(item)
        
        for instruction, option_map, perm_idx in perm_results:
            
            # [CRITICAL] Gemma-3n message structure
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"}, # Placeholder, vLLM will automatically match multi_modal_data
                        {"type": "text", "text": instruction}
                    ]
                }
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
                    # Pass padded audio
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
        exp_id = f"parade_gemma_{timestamp}"
        if args.limit: exp_id += "_test"

    # PARADE results are not separated by dimension folder, placed directly under exp_id
    exp_dir = os.path.join(RESULTS_ROOT, exp_id)
    ensure_dir(exp_dir) 
    print(f"Results output directory: {exp_dir}")

    for model_path in MODELS_TO_RUN:
        model_short_name = model_path.split("/")[-1]
        print(f"\n{'='*40}")
        print(f"Preparing to run model: {model_short_name}")
        print(f"{'='*40}")

        # 2.1 Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            continue

        # 2.2 Initialize vLLM
        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=4096, 
                gpu_memory_utilization=0.9, 
                limit_mm_per_prompt={"audio": 1}
            )
        except Exception as e:
            print(f"\n[Fatal Error] Initialization failed: {e}")
            continue

        # Set sampling parameters
        stop_token_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "additional_special_tokens_ids"):
            stop_token_ids.extend(tokenizer.additional_special_tokens_ids)

        sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=0.8,
            max_tokens=128, 
            stop_token_ids=stop_token_ids
        )

        output_jsonl = os.path.join(exp_dir, f"PARADE_{model_short_name}_generations.jsonl")
        
        processed_files = get_processed_files(output_jsonl)
        items_to_run = [d for d in dataset if os.path.basename(d["audio_path"]) not in processed_files]
        
        if not items_to_run:
            print(f"All completed, skipping.")
            continue
            
        print(f"\nRemaining {len(items_to_run)} raw data items")

        chunk_size = BATCH_SIZE
        
        for i in tqdm(range(0, len(items_to_run), chunk_size), desc=f"{model_short_name}"):
            batch_items = items_to_run[i : i + chunk_size]
            
            try:
                # Prepare inputs (SR=24000 for Gemma)
                prepared_data = prepare_gemma_inputs_parade(batch_items, tokenizer, target_sr=24000)
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