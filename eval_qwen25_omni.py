import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import json
import glob
import librosa
import datetime
import numpy as np
import argparse
import gc
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths, get_audio_id_from_path
from transformers import Qwen2_5OmniProcessor

# ================= Configuration =================
AUDIO_DIR = "/mnt/data/ycevan/datasets/CREMA-D/Audios"
RESULTS_ROOT = "./results"

MODELS_TO_RUN = [
    "Qwen/Qwen2.5-Omni-7B",
    "Qwen/Qwen2.5-Omni-3B"
]

# ================= Utility Functions =================
def prepare_vllm_inputs(audio_paths, instruction, processor, target_sr=16000):
    """
    Prepare vLLM inputs, including:
    1. Loadingaudio (16k for Qwen)
    2. Dynamic padding within batch
    3. Generate prompt template with tokenizer
    """
    inputs = []
    raw_audios = []
    valid_paths = []
    
    # --- Phase 1: Read and filter ---
    for path in audio_paths:
        try:
            # Qwen-Audio/Omni series typically uses 16000 Hz
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            raw_audios.append(y)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading audio {path}: {e}")
            continue
    
    if not raw_audios:
        return []

    # --- Phase 2: Calculate max length and apply padding ---
    # Find the longest audio length in this batch
    max_len = max([len(y) for y in raw_audios])
    
    padded_audios = []
    for y in raw_audios:
        if len(y) < max_len:
            # Pad with 0 (silence) at the end
            # np.pad(array, (pad_before, pad_after), mode)
            y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            padded_audios.append(y_padded)
        else:
            padded_audios.append(y)

    # --- Phase 3: Build vLLM inputs and prompts ---
    for path, y_padded in zip(valid_paths, padded_audios):
        # [Key fix] Qwen in vLLM recommends using string format directly, not list of dicts
        # Manually insert audio token before the instruction
        content_str = f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{instruction}"

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            },
            {
                "role": "user",
                "content": content_str  # Pass string here
            }
        ]

        # [Key fix] Use processor.tokenizer to apply template
        # This avoids processor trying to parse "audio_url" or validate placeholder causing errors
        try:
            final_prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Template apply failed: {e}, using manual fallback.")
            # Safety fallback
            final_prompt = (
                "<|im_start|>system\nYou are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.<|im_end|>\n"
                f"<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        inputs.append({
            "prompt": final_prompt,
            "multi_modal_data": {
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
    parser.add_argument("-e", "--exp_id", type=str, default="vllm_omni_test", help="Experiment ID (folder name)")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Inference batch size")
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
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp}"
        if args.limit: exp_id += "_test"

    exp_dir = os.path.join(RESULTS_ROOT, exp_id, args.dimension)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results output directory: {exp_dir}")

    # 2. Loop over models
    for model_path in MODELS_TO_RUN:
        model_short_name = model_path.split("/")[-1] # e.g. "Qwen2.5-Omni-7B"
        print(f"\n{'='*40}")
        print(f"Preparing to run model: {model_short_name}")
        print(f"{'='*40}")
        
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

        # 2.1 Initialize vLLM
        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.95, # May need to lower if running other processes
                limit_mm_per_prompt={"audio": 1}
            )
        except Exception as e:
            print(f"\n[Fatal Error] model {model_path} Initialization failed: {e}")
            continue

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.8,
            max_tokens=512,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        # 2.2 Run tasks
        for task_name, task_info in TASKS.items():
            instruction = task_info["instruction"]
            # Filename includes specific model name
            output_jsonl = os.path.join(exp_dir, f"{task_name}_{model_short_name}_generations.jsonl")
            
            # Resume mechanism
            processed_files = get_processed_files(output_jsonl)
            files_to_run = [
                f for f in sorted_files 
                if get_audio_id_from_path(f, args.dimension) not in processed_files
            ]
            
            if not files_to_run:
                print(f"Task [{task_name}] completed, skipping.")
                continue
                
            print(f"\nTask [{task_name}] - Remaining {len(files_to_run)} items")

            # Batch execution
            chunk_size = args.batch_size
            
            for i in tqdm(range(0, len(files_to_run), chunk_size), desc=f"{model_short_name} | {task_name}"):
                batch_files = files_to_run[i : i + chunk_size]
                
                try:
                    inputs_list = prepare_vllm_inputs(batch_files, instruction, processor)
                    if not inputs_list: continue

                    # vLLM Generate
                    outputs = llm.generate(inputs_list, sampling_params, use_tqdm=False)
                    
                    # Write results
                    with open(output_jsonl, "a", encoding="utf-8") as f:
                        for input_item, output_item in zip(inputs_list, outputs):
                            path = input_item["meta_path"]
                            generated_text = output_item.outputs[0].text.strip()
                            
                            audio_id = get_audio_id_from_path(path, args.dimension)
                            
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
                    continue

        # 2.3 Free memory for the next model
        print(f"model {model_short_name} execution completed, releasing memory...")
        del llm
        cleanup_vllm()

    print("\nAll models and tasks completed.")

if __name__ == "__main__":
    main()