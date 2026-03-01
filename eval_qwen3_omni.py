import os
import json
import glob
import librosa
import argparse
import torch
import gc
from tqdm import tqdm
import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Import shared configuration
from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths

# ================= Configuration =================
AUDIO_DIR = "/mnt/data/ycevan/datasets/CREMA-D/Audios"
RESULTS_ROOT = "./results"

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# ================= Utility Functions =================

def prepare_qwen3_inputs(audio_paths, instruction, tokenizer, target_sr=16000):
    """
    Prepare Qwen3-Omni vLLM input format.
    Uses Tokenizer's Chat Template to ensure correct format.
    """
    inputs = []
    
    for path in audio_paths:
        try:
            # 1. Load audio (Qwen series typically uses 16k)
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            
            # 2. Build conversation messages
            # Qwen3-Omni supports standard multimodal message format
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": path}, # audio_url is just a marker; actual data passed via multi_modal_data
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            
            # 3. Apply Chat Template to generate prompt string
            # This auto-inserts <|audio_bos|><|AUDIO|><|audio_eos|> tokens
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": (y, sr) # vLLM accepts (data, sr) tuple
                },
                "meta_path": path 
            })
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
        
    return inputs

# ================= Main Program =================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit the number of test samples")
    parser.add_argument("-e", "--exp_id", type=str, default=None, help="Experiment ID")
    parser.add_argument("--tp", type=int, default=2, help="Tensor Parallel Size (number of GPUs), recommended 2 or 4 for 30B model")
    parser.add_argument("-b","--batch_size", type=int, default=8, help="Inference batch size")
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

    # 2. Initialize Tokenizer (for processing Prompt Template)
    print(f"Loading Tokenizer: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 3. Initialize vLLM
    print(f"Initializing vLLM model: {MODEL_ID} (TP={args.tp}) ...")
    
    try:
        llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            max_model_len=8192,           
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"audio": 1},
        )
    except Exception as e:
        print(f"\n[Fatal Error] vLLM Initialization failed: {e}")
        return

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.8,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    )

    # 4. Execute Task
    model_short_name = "Qwen3-Omni-30B-A3B-Instruct"

    for task_name, task_info in TASKS.items():
        instruction = task_info["instruction"]
        output_jsonl = os.path.join(exp_dir, f"{task_name}_{model_short_name}_generations.jsonl")
        
        # Resume mechanism
        processed_files = get_processed_files(output_jsonl)
        files_to_run = [f for f in audio_files if os.path.basename(f) not in processed_files]
        
        if not files_to_run:
            print(f"Task [{task_name}] completed, skipping.")
            continue
            
        print(f"\nTask [{task_name}] - Remaining {len(files_to_run)} items")

        chunk_size = args.batch_size
        
        for i in tqdm(range(0, len(files_to_run), chunk_size), desc=f"{task_name}"):
            batch_files = files_to_run[i : i + chunk_size]
            
            try:
                # Prepare inputs
                inputs_list = prepare_qwen3_inputs(batch_files, instruction, tokenizer)
                if not inputs_list: continue

                # Run inference
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
                torch.cuda.empty_cache()
                continue

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()