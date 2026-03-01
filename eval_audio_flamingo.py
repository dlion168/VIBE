"""
Evaluation Script for Nvidia Audio-Flamingo-3 (AF3) using vLLM
Fix: Use librosa to load audio for llm.generate (since generate API doesn't support URLs)
"""
import os
# [CRITICAL] Must be set before importing vllm
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# AF3 needs larger context; allow vLLM to auto-adjust
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import json
import glob
import librosa
import numpy as np
import argparse
import torch
import gc
import datetime
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Import shared configuration
from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths, get_audio_id_from_path

# ================= Configuration =================
RESULTS_ROOT = "./results"

MODEL_ID = "nvidia/audio-flamingo-3-hf"

# ================= Utility Functions =================

def prepare_vllm_inputs(audio_paths, instruction, target_sr=16000):
    """
    Prepare AF3 inputs:
    1. [FIX] Use librosa to load audio as numpy array (llm.generate does not support URLs)
    2. Use ChatML format + <sound> placeholder
    """
    inputs = []
    valid_paths = []
    
    # Use tqdm to show progress
    for path in tqdm(audio_paths, desc="Loading Audio & Preparing Prompts"):
        try:
            # [CRITICAL FIX] llm.generate requires raw audio data (numpy array)
            # AF3 default sample rate is 16000
            y, sr = librosa.load(path, sr=target_sr, mono=True)
        except Exception as e:
            print(f"Error loading audio {path}: {e}")
            continue

        # Manually build ChatML-format prompt (avoid tokenizer issues)
        final_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<sound>{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs.append({
            "prompt": final_prompt,
            "multi_modal_data": {
                # vLLM generate interface accepts (numpy_array, sample_rate) tuple
                "audio": (y, sr) 
            },
            "meta_path": path 
        })
        valid_paths.append(path)
        
    return inputs, valid_paths

def cleanup_vllm():
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU Memory released.")

# ================= Main Program =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of test samples")
    parser.add_argument("-e", "--exp_id", type=str, default="20260124_013612", help="Experiment ID")
    parser.add_argument("-d", "--dimension", type=str, default="gender", choices=["gender", "accent"], help="Select evaluation dimension (gender/accent)")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Prepare files (read from task_config)
    print(f"Loading dataset，dimension: {args.dimension}")
    try:
        audio_files = get_audio_paths(args.dimension)
    except Exception as e:
        print(f"Failed to read paths: {e}")
        return
    
    if not audio_files:
        print(f"Error: in dimension {args.dimension}  no audio files found.")
        return
    
    if args.limit:
        print(f"[Test mode] Using only the first {args.limit} items")
        audio_files = audio_files[:args.limit]

    # Optimization: pre-scan audio lengths and sort
    print("Scanning audio file lengths for processing optimization...")
    file_lengths = []
    for f in tqdm(audio_files, desc="Scanning Lengths"):
        try:
            duration = librosa.get_duration(path=f)
            file_lengths.append((f, duration))
        except:
            continue
    
    file_lengths.sort(key=lambda x: x[1])
    sorted_files = [x[0] for x in file_lengths]
    print(f"Sorted {len(sorted_files)} files by length.")

    # configurationdirectory
    if args.exp_id:
        exp_id = args.exp_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"vllm_af3_{timestamp}"
        if args.limit: exp_id += "_test"

    exp_dir = os.path.join(RESULTS_ROOT, exp_id, args.dimension)
    ensure_dir(exp_dir)
    print(f"Results output directory: {exp_dir}")

    # 2. Run AF3 model
    model_short_name = MODEL_ID.split("/")[-1]
    print(f"\n{'='*40}")
    print(f"Preparing to run model: {model_short_name}")
    print(f"{'='*40}")

    # 2.2 Initialize vLLM
    try:
        llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.75,
            limit_mm_per_prompt={"audio": 1},
            enforce_eager=True, 
            # allowed_local_media_path="/"  <-- not needed for llm.generate since we pass raw data
        )
    except Exception as e:
        print(f"\n[Fatal Error] vLLM Initialization failed: {e}")
        return

    # ChatML format stop tokens
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=stop_tokens
    )

    # 3. Execute tasks
    for task_name, task_info in TASKS.items():
        instruction = task_info["instruction"]
        output_jsonl = os.path.join(exp_dir, f"{task_name}_{model_short_name}_generations.jsonl")
        
        processed_files = get_processed_files(output_jsonl)
        files_to_run = [f for f in sorted_files if get_audio_id_from_path(f, args.dimension) not in processed_files]
        
        if not files_to_run:
            print(f"Task [{task_name}] completed, skipping.")
            continue
            
        print(f"\nTask [{task_name}] - Remaining {len(files_to_run)} items")

        # Prepare all inputs
        inputs_list, valid_paths = prepare_vllm_inputs(files_to_run, instruction)
        
        if not inputs_list: 
            print("No valid inputs prepared.")
            continue

        print(f"Starting vLLM generate for {len(inputs_list)} items...")
        
        try:
            # Use llm.generate + raw data
            outputs = llm.generate(inputs_list, sampling_params, use_tqdm=True)
            
            # Write results
            with open(output_jsonl, "a", encoding="utf-8") as f:
                for path, output_item in zip(valid_paths, outputs):
                    generated_text = output_item.outputs[0].text.strip()
                    
                    # audio_file field logic
                    filename = os.path.basename(path)
                    
                    if args.dimension == "accent":
                        parent_dir = os.path.basename(os.path.dirname(path))
                        parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
                        audio_id = os.path.join(parent_parent_dir, parent_dir, filename)
                    else:
                        audio_id = filename
                    
                    record = {
                        "audio_file": audio_id,
                        "model": model_short_name,
                        "task": task_name,
                        "prompt": instruction,
                        "response": generated_text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            torch.cuda.empty_cache()
            continue

    del llm
    cleanup_vllm()

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()