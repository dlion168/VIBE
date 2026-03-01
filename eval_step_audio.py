"""
Evaluation Script for Step-Audio-2 Series using vLLM (Offline Inference)
Improved with Base64 Audio Encoding and Chat Interface
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
import torch
import gc
import datetime
from tqdm import tqdm

# vLLM Imports
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Import shared configuration
from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths

# ================= Configuration =================
RESULTS_ROOT = "./results"

MODELS_TO_RUN = [
    #"stepfun-ai/Step-Audio-2-mini",
    #"stepfun-ai/Step-Audio-2-mini-Base",
    "stepfun-ai/Step-Audio-R1"
]

# vLLM can use a larger batch size
BATCH_SIZE = 16

# ================= Utility Functions =================

def get_audio_id_from_path(path, dimension):
    filename = os.path.basename(path)
    if dimension == "accent":
        parent_dir = os.path.basename(os.path.dirname(path))
        parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return os.path.join(parent_parent_dir, parent_dir, filename)
    else:
        return filename

def encode_audio_to_base64(audio_path, target_sr=16000):
    """
    Read audio and convert to Base64-encoded WAV format (Int16 PCM)
    Based on StepAudio2 official processing logic
    """
    try:
        # 1. Load and Resample
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 2. Convert to Int16 PCM
        # Clip to avoid overflow and scale to 16-bit integer range
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

def prepare_chat_inputs(audio_paths, instruction):
    """
    Prepare a list of messages for the vLLM chat interface
    """
    conversations = []
    valid_paths = []
    
    for path in audio_paths:
        # Encode audio
        audio_b64 = encode_audio_to_base64(path)
        if audio_b64 is None:
            continue
            
        # Construct message in Step-Audio/OpenAI format
        # Use "input_audio" type, the standard way vLLM handles OpenAI-compatible inputs
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
            },
            {"role": "assistant", "content": None}
        ]
        
        conversations.append(messages)
        valid_paths.append(path)
        
    return conversations, valid_paths

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
    parser.add_argument("-d", "--dimension", type=str, default="gender", choices=["gender", "accent"], help="Select evaluation dimension")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch Size")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Prepare files
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

    # Simple sorting by length (uses librosa for quick duration scan; use os.path.getsize if too slow)
    print("Scanning audio file lengths for batch optimization...")
    file_lengths = []
    for f in tqdm(audio_files, desc="Scanning Lengths"):
        try:
            d = librosa.get_duration(path=f)
            file_lengths.append((f, d))
        except:
            continue
    file_lengths.sort(key=lambda x: x[1])
    sorted_files = [x[0] for x in file_lengths]
    print(f"Sorted {len(sorted_files)} files by length.")

    if args.exp_id:
        exp_id = args.exp_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"step_audio_2_vllm_chat_{timestamp}"
        if args.limit: exp_id += "_test"

    exp_dir = os.path.join(RESULTS_ROOT, exp_id, args.dimension)
    ensure_dir(exp_dir)
    print(f"Results output directory: {exp_dir}")

    # 2. Loop over models
    for model_path in MODELS_TO_RUN:
        model_short_name = model_path.split("/")[-1]
        print(f"\n{'='*40}")
        print(f"Preparing to run model (vLLM Chat): {model_short_name}")
        print(f"{'='*40}")

        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.9,
                limit_mm_per_prompt={"input_audio": 1} # Note: changed to input_audio here
            )
            # vLLM's tokenizer doesn't need manual retrieval; llm.chat uses it automatically
        except Exception as e:
            print(f"\n[Fatal Error] vLLM Initialization failed: {e}")
            continue

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096
        )

        # 3. Execute tasks
        for task_name, task_info in TASKS.items():
            instruction = task_info["instruction"]
            output_jsonl = os.path.join(exp_dir, f"{task_name}_{model_short_name}_generations.jsonl")
            
            processed_files = get_processed_files(output_jsonl)
            
            files_to_run = [
                f for f in sorted_files 
                if get_audio_id_from_path(f, args.dimension) not in processed_files
            ]
            
            if not files_to_run:
                print(f"Task [{task_name}] completed, skipping.")
                continue
                
            print(f"\nTask [{task_name}] - Remaining {len(files_to_run)} items")

            chunk_size = args.batch_size
            
            for i in tqdm(range(0, len(files_to_run), chunk_size), desc=f"{model_short_name} | {task_name}"):
                batch_files = files_to_run[i : i + chunk_size]
                
                try:
                    # Prepare vLLM Chat inputs (with Base64 Audio)
                    conversations, batch_paths = prepare_chat_inputs(batch_files, instruction)
                    
                    if not conversations: 
                        continue

                    # Execute Chat generation
                    # use_tqdm=False to prevent inner progress bar from interfering with outer one
                    outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
                    
                    # Write results
                    with open(output_jsonl, "a", encoding="utf-8") as f:
                        for path, output_item in zip(batch_paths, outputs):
                            # vLLM Chat Output structure: output_item.outputs[0].text
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
                    torch.cuda.empty_cache()
                    continue

        del llm
        cleanup_vllm()

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()