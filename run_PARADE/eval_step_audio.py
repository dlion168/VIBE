"""
Evaluation Script for PARADE Dataset using Step-Audio-2 with vLLM
Features:
1. Full Permutation (6 prompts per audio) to eliminate positional bias.
2. Step-Audio-2 specific Base64 audio input format (16kHz WAV).
3. vLLM batched inference with raw audio data.
"""
import os
# [CRITICAL] Must be set before importing vllm
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import io
import wave
import base64
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
    "stepfun-ai/Step-Audio-2-mini",
    "stepfun-ai/Step-Audio-2-mini-Base" 
]

# 16 data items * 6 permutations = 96 vLLM Requests
BATCH_SIZE = 16

# ================= Utility Functions =================

def encode_audio_to_base64(audio_path, target_sr=16000):
    """
    Read audio and convert to Base64-encoded WAV format (Int16 PCM)
    """
    try:
        # 1. Load and Resample
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 2. Convert to Int16 PCM
        y_int16 = (y.clip(-1.0, 1.0) * 32767.0).astype('int16')
        
        # 3. Write to BytesIO as WAV
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)      # Mono
            wf.setsampwidth(2)      # 2 bytes (16 bits)
            wf.setframerate(target_sr)
            wf.writeframes(y_int16.tobytes())
            
        # 4. Encode to Base64
        base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return base64_str
        
    except Exception as e:
        print(f"[Error] Failed to encode audio {audio_path}: {e}")
        return None

def prepare_stepaudio_inputs_parade(batch_items, target_sr=16000):
    """
    Prepare Step-Audio-2 vLLM inputs (PARADE version)
    Features:
    1. Audio Encoding (Base64 WAV)
    2. Chat Message Construction
    3. Full Permutation of Options
    """
    conversations = []
    valid_metas = []
    
    # Pre-process audio encoding (Cache audio to avoid re-encoding for 6 permutations)
    audio_cache = {} # path -> base64_str
    
    for item in batch_items:
        path = item["audio_path"]
        if path not in audio_cache:
            b64_str = encode_audio_to_base64(path, target_sr)
            if b64_str:
                audio_cache[path] = b64_str
            else:
                print(f"Skipping {path} due to encoding failure.")

    # Generate inputs
    for item in batch_items:
        path = item["audio_path"]
        if path not in audio_cache:
            continue
            
        audio_b64 = audio_cache[path]
        
        # Get 6 permutations (from task_config)
        perm_results = generate_parade_permutations(item)
        
        for instruction, option_map, perm_idx in perm_results:
            
            # Step-Audio / OpenAI Compatible Message Structure
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio", 
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav"
                            }
                        },
                        {
                            "type": "text", 
                            "text": instruction
                        }
                    ]
                }
            ]
            
            conversations.append(messages)
            
            valid_metas.append({
                "path": item["audio_path"],
                "option_map": option_map, # Key: record A/B/C to text mapping
                "instruction": instruction,
                "label": item["label"],
                "speaker_model": item["speaker_model"],
                "domain": item["domain"],
                "perm_idx": perm_idx
            })
        
    return conversations, valid_metas

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
        exp_id = f"parade_stepaudio_{timestamp}"
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

        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.9,
                limit_mm_per_prompt={"input_audio": 1}, # Step-Audio uses input_audio
            )
        except Exception as e:
            print(f"\n[Fatal Error] Initialization failed: {e}")
            continue

        sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=0.8,
            max_tokens=128, 
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
                # Prepare inputs (SR=16000 for Step-Audio)
                conversations, metas = prepare_stepaudio_inputs_parade(batch_items, target_sr=16000)
                if not conversations: continue
                
                # vLLM Chat Generate
                outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
                
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
        cleanup_vllm()

    print("\nDone.")

if __name__ == "__main__":
    main()