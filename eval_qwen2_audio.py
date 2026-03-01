import os
# [CRITICAL] Must be set before importing vllm to avoid CUDA initialization failures
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import glob
import librosa
import numpy as np
import argparse
import gc
import torch
import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer

# Import shared configuration
from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths

# ================= Configuration =================

AUDIO_DIR = "/mnt/data/ycevan/datasets/CREMA-D/Audios"
RESULTS_ROOT = "./results"

# Model list
MODELS_TO_RUN = [
    "Qwen/Qwen2-Audio-7B-Instruct"
]

# Batch size
BATCH_SIZE = 32

# ================= Utility Functions =================

def prepare_qwen_inputs(audio_paths, instruction, tokenizer, target_sr=16000):
    """
    Prepare Qwen2-Audio vLLM inputs
    [Fix] Changed content to string format to fix template generation failed issue
    """
    inputs = []
    raw_audios = []
    valid_paths = []
    
    # --- Phase 1: Read and filter ---
    for path in audio_paths:
        try:
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            raw_audios.append(y)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading audio {path}: {e}")
            continue
    
    if not raw_audios:
        return []

    # --- Phase 2: Calculate max length and apply padding ---
    max_len = max([len(y) for y in raw_audios])
    
    padded_audios = []
    for y in raw_audios:
        if len(y) < max_len:
            y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            padded_audios.append(y_padded)
        else:
            padded_audios.append(y)

    # --- Phase 3: Build vLLM inputs and prompts ---
    for path, y_padded in zip(valid_paths, padded_audios):
        content_str = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{instruction}"
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": content_str
            }
        ]

        try:
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Template generation failed for {path}: {e}")
            continue

        inputs.append({
            "prompt": final_prompt,
            "multi_modal_data": {
                # vLLM maps <|AUDIO|> to the tensor passed here
                "audio": (y_padded, target_sr) 
            },
            "meta_path": path 
        })
        
    return inputs

def cleanup_vllm():
    """Force release GPU memory occupied by vLLM"""
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU Memory released.")

# ================= Main Program =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of test samples (for debugging)")
    parser.add_argument("-e", "--exp_id", type=str, default=None, help="Experiment ID (folder name)")
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
        print(f"[Test mode] Using only the first {args.limit} audio files")
        audio_files = audio_files[:args.limit]

    # [Optimization] Pre-scan audio lengths and sort (Sort by Length)
    print("Scanning audio file lengths for batch optimization (reduce padding waste)...")
    file_lengths = []
    for f in tqdm(audio_files, desc="Scanning Lengths"):
        try:
            # get_duration only reads file header, very fast
            duration = librosa.get_duration(path=f)
            file_lengths.append((f, duration))
        except:
            continue
    
    # Sort by length (shortest to longest)
    file_lengths.sort(key=lambda x: x[1])
    sorted_files = [x[0] for x in file_lengths]
    print(f"Sorted {len(sorted_files)} files by length.")

    # Set output directory
    if args.exp_id:
        exp_id = args.exp_id
        print(f"[Resume mode] Continuing experiment: {exp_id}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = timestamp
        if args.limit:
            exp_id += "_test"
        print(f"[New mode] Creating new experiment: {exp_id}")

    exp_dir = os.path.join(RESULTS_ROOT, exp_id, args.dimension)
    ensure_dir(exp_dir) 
    print(f"Results output directory: {exp_dir}")

    # 2. Loop over models
    for model_path in MODELS_TO_RUN:
        model_short_name = model_path.split("/")[-1]
        print(f"\n{'='*40}")
        print(f"Preparing to run model: {model_short_name}")
        print(f"{'='*40}")

        # 2.1 Load Tokenizer
        print(f"Loading Tokenizer: {model_path} ...")
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
            print(f"\n[Fatal Error] model {model_path} Initialization failed: {e}")
            continue

        # Set sampling parameters
        stop_token_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "additional_special_tokens_ids"):
            stop_token_ids.extend(tokenizer.additional_special_tokens_ids)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=stop_token_ids
        )

        # 2.3 Execute tasks
        for task_name, task_info in TASKS.items():
            instruction = task_info["instruction"]
            output_jsonl = os.path.join(exp_dir, f"{task_name}_{model_short_name}_generations.jsonl")
            
            # Resume mechanism
            processed_files = get_processed_files(output_jsonl)
            files_to_run = [f for f in sorted_files if os.path.basename(f) not in processed_files]
            
            if not files_to_run:
                print(f"Task [{task_name}] completed, skipping.")
                continue
                
            print(f"\nTask [{task_name}] - Remaining {len(files_to_run)} items")

            chunk_size = BATCH_SIZE
            
            for i in tqdm(range(0, len(files_to_run), chunk_size), desc=f"{model_short_name} | {task_name}"):
                batch_files = files_to_run[i : i + chunk_size]
                
                try:
                    # Prepare inputs
                    inputs_list = prepare_qwen_inputs(batch_files, instruction, tokenizer, target_sr=16000)
                    if not inputs_list: continue

                    # vLLM Generate
                    outputs = llm.generate(inputs_list, sampling_params, use_tqdm=False)
                    
                    # Write results
                    with open(output_jsonl, "a", encoding="utf-8") as f:
                        for input_item, output_item in zip(inputs_list, outputs):
                            path = input_item["meta_path"]
                            generated_text = output_item.outputs[0].text.strip()
                            
                            # [MODIFIED] audio_file field logic
                            filename = os.path.basename(path)
                            
                            if args.dimension == "accent":
                                # If accent, include parent directories (e.g. "wav/file.wav")
                                parent_dir = os.path.basename(os.path.dirname(path))
                                parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
                                audio_id = os.path.join(parent_parent_dir, parent_dir, filename)
                            else:
                                # If gender (or other), keep only the filename (e.g. "file.wav")
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
                    print(f"[Error] Batch failed: {e}")
                    # Try to free GPU memory
                    torch.cuda.empty_cache()
                    continue

        # 2.4 Release memory
        print(f"model {model_short_name} execution completed, releasing memory...")
        del llm
        del tokenizer
        cleanup_vllm()

    print("\nAll models and tasks completed.")

if __name__ == "__main__":
    main()