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
from tqdm import tqdm

from huggingface_hub import snapshot_download  # [Added] For downloading model to get LoRA path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest      # [Added] For mounting Speech LoRA
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer

# Import shared configuration
try:
    from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths
except ImportError:
    TASKS = {"Speech_Description": {"instruction": "Describe the audio in detail."}}
    def get_processed_files(f): return set()
    def ensure_dir(d): os.makedirs(d, exist_ok=True)

# ================= Configuration =================
RESULTS_ROOT = "./results"

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
BATCH_SIZE = 32
# ================= Utility Functions =================

def prepare_phi4_inputs(audio_paths, instruction, target_sr=16000):
    """
    Prepare Phi-4 Multimodal inputs
    [Fix 1] Use <|audio_1|> as placeholder (per official docs).
    """
    inputs = []
    raw_audios = []
    valid_paths = []
    
    # --- 1. Load audio ---
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

    # --- 2. Dynamic Padding ---
    max_len = max([len(y) for y in raw_audios])
    padded_audios = []
    for y in raw_audios:
        if len(y) < max_len:
            y_padded = np.pad(y, (0, max_len - len(y)), mode='constant')
            padded_audios.append(y_padded)
        else:
            padded_audios.append(y)

    # --- 3. Constructing Prompt String ---
    # [Fix 1] Official format: <|user|><|audio_1|>{instruction}<|end|><|assistant|>
    # Note: it must be <|audio_1|> here, and a newline before instruction is optional depending on prompt style
    prompt_text = f"<|user|><|audio_1|>{instruction}<|end|><|assistant|>"

    for path, y_padded in zip(valid_paths, padded_audios):
        inputs.append({
            "prompt": prompt_text,
            "multi_modal_data": {
                "audio": (y_padded, target_sr) 
            },
            "meta_path": path 
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

    # Sort by length
    print("Scanning audio file lengths for batch optimization...")
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

    # Set output directory
    if args.exp_id:
        exp_id = args.exp_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"vllm_phi4_{timestamp}"
        if args.limit: exp_id += "_test"

    exp_dir = os.path.join(RESULTS_ROOT, exp_id, args.dimension)
    ensure_dir(exp_dir)
    print(f"Results output directory: {exp_dir}")

    # 2. Initialize model and LoRA
    model_short_name = MODEL_ID.split("/")[-1]
    print(f"\n{'='*40}")
    print(f"Preparing to run model: {model_short_name}")
    print(f"{'='*40}")

    print("Checking/downloading model Snapshot...")
    model_path = snapshot_download(MODEL_ID)
    speech_lora_path = os.path.join(model_path, "speech-lora")
    print(f"Speech LoRA path: {speech_lora_path}")

    # 2.2 Initialize vLLM (enable LoRA)
    print("Initializing vLLM (Enable LoRA)...")
    try:
        llm = LLM(
            model=model_path, # Pass in local path
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"audio": 1},
            # [Fix 3] Enable LoRA configuration
            enable_lora=True,
            max_lora_rank=320, 
            enforce_eager=True,
        )
    except Exception as e:
        print(f"\n[Fatal Error] vLLM Initialization failed: {e}")
        return

    # [Fix 4] Create Speech LoRA Request
    # 'speech' is the adapter name, id=1 is the adapter id, path points to the downloaded speech-lora
    speech_lora_req = LoRARequest("speech", 1, speech_lora_path)

    # Set sampling parameters
    stop_tokens = ["<|end|>", "<|endoftext|>"]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=stop_tokens,
    )

    # 3. Execute tasks
    for task_name, task_info in TASKS.items():
        instruction = task_info["instruction"]
        output_jsonl = os.path.join(exp_dir, f"{task_name}_{model_short_name}_generations.jsonl")
        
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
                # Prepare inputs (No need to input tokenizer or audio_token_id)
                inputs_list = prepare_phi4_inputs(batch_files, instruction, target_sr=16000)
                if not inputs_list: continue

                # [Fix 5] Pass lora_request during generation
                outputs = llm.generate(
                    inputs_list, 
                    sampling_params, 
                    use_tqdm=False,
                    lora_request=speech_lora_req 
                )
                
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
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

    del llm
    cleanup_vllm()
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()