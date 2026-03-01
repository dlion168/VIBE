"""
Evaluation Script for Gemini Batch API (Cost Saving Mode)
Replaces vLLM with Google Generative AI SDK using Async Batch Jobs.

Usage:
1. Submit Jobs: python script.py --mode submit -d gender
2. Get Results: python script.py --mode retrieve -d gender
"""
import os
import json
import time
import argparse
import glob
import datetime
from tqdm import tqdm

from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
# Import shared configuration
try:
    from task_config import TASKS, get_processed_files, ensure_dir, get_audio_paths
except ImportError:
    # Fallback for standalone testing
    print("[Warning] task_config not found, using dummy config.")
    TASKS = {"Test": {"instruction": "Transcribe this."}}
    def get_processed_files(f): return set()
    def ensure_dir(d): os.makedirs(d, exist_ok=True)
    def get_audio_paths(d): return []

# ================= Configuration =================

RESULTS_ROOT = "./results"
CACHE_FILE = "gemini_file_cache.json"  # Cache uploaded file URIs to avoid wasting money/time on re-uploads

# Gemini model (models supporting Batch Mode)
# Recommend using Flash for best cost-performance, or Pro for strongest capability
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025" 

# ================= Utility Functions =================

def load_file_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_file_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def upload_single_file(path, client):
    """Single file upload and status check logic"""
    try:
        g_file = client.files.upload(
            file=path,
            config={'mime_type': 'audio/wav'}    
        )
        # Wait for processing to complete
        while g_file.state.name == 'PROCESSING':
            time.sleep(2)
            g_file = client.files.get(name=g_file.name)
        
        if g_file.state.name == 'FAILED':
            return path, None
        return path, g_file.uri
    except Exception as e:
        print(f"Error uploading {path}: {e}")
        return path, None

def upload_audio_files(audio_paths, client, max_workers=10):
    cache = load_file_cache()
    uris = {}
    files_to_upload = []

    # 1. Filter files that need uploading
    for path in audio_paths:
        abs_path = os.path.abspath(path)
        if abs_path in cache:
            uris[path] = cache[abs_path]
        else:
            files_to_upload.append(path)

    if not files_to_upload:
        return uris

    # 2. Upload in parallel
    print(f"Parallel uploading {len(files_to_upload)} files with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(upload_single_file, p, client): p for p in files_to_upload}
        
        for future in tqdm(as_completed(future_to_path), total=len(files_to_upload), desc="Uploading"):
            path, uri = future.result()
            if uri:
                uris[path] = uri
                cache[os.path.abspath(path)] = uri

    save_file_cache(cache)
    return uris

def create_batch_job(model_name, tasks_to_run, audio_files, file_uris, dimension, client):
    """
    Create and submit Batch Job
    """
    batch_requests = []
    print(f"Constructing batch requests for: {list(tasks_to_run.keys())}")
    
    for task_name, task_info in tasks_to_run.items():
        instruction = task_info["instruction"]
        
        for audio_path in audio_files:
            if audio_path not in file_uris:
                continue
                
            file_uri = file_uris[audio_path]
            filename = os.path.basename(audio_path)
            custom_id = f"{task_name}|{dimension}|{filename}"
            
            # Package in OpenAI format
            request = {
                "key": custom_id,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"file_data": {"mime_type": "audio/wav", "file_uri": file_uri}},
                                {"text": instruction}
                            ]
                        }
                    ],
                    "generation_config": {
                        "temperature": 0.0,
                        "max_output_tokens": 4096,
                        'response_modalities': ['TEXT']
                    }
                }
            }
            batch_requests.append(request)

    if not batch_requests:
        print("No valid requests.")
        return None

    batch_input_filename = f"batch_input_{dimension}_{int(time.time())}.jsonl"
    with open(batch_input_filename, 'w', encoding='utf-8') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Uploading request file: {batch_input_filename}")
    input_file_ref = client.files.upload(
        file=batch_input_filename,
        config={'mime_type': 'application/jsonl'}    
    )
    
    while input_file_ref.state.name == 'PROCESSING':
        time.sleep(2)
        input_file_ref = client.files.get(name=input_file_ref.name)

    print("Submitting Batch Job...")
    batch_job = client.batches.create(
        model=model_name,
        src=input_file_ref.name,
    )
    
    print(f"Batch Job Created! Name: {batch_job.name}")
    
    with open("gemini_jobs.jsonl", "a") as f:
        job_record = {
            "job_id": batch_job.name,
            "dimension": dimension,
            "status": "SUBMITTED",
            "timestamp": datetime.datetime.now().isoformat()
        }
        f.write(json.dumps(job_record) + "\n")
        
    os.remove(batch_input_filename)
    return batch_job.name

def retrieve_and_auto_parse(client, exp_dir):
    """Check status, auto-download and parse if successful"""
    if not os.path.exists("gemini_jobs.jsonl"): return

    with open("gemini_jobs.jsonl", "r") as f:
        jobs = [json.loads(line) for line in f if line.strip()]
            
    updated = False
    for job in jobs:
        if job["status"] in ["COMPLETED", "FAILED"]: continue
            
        try:
            b_job = client.batches.get(name=job["job_id"])
            state = b_job.state.name
            print(f"Job {job['job_id']}: {state}")
            
            if state == 'JOB_STATE_SUCCEEDED':
                # 1. Download results
                raw_file_name = b_job.dest.file_name
                print(f"  Downloading results from {raw_file_name}...")
                content = client.files.download(file=raw_file_name)
                
                # 2. Save raw backup
                raw_dir = os.path.join(exp_dir, "gemini_raw_results")
                ensure_dir(raw_dir)
                raw_path = os.path.join(raw_dir, f"{job['job_id'].split('/')[-1]}.jsonl")
                with open(raw_path, "wb") as f: f.write(content)
                
                # 3. Auto-parse
                parse_and_save_content(content, job.get("model", MODEL_NAME), exp_dir)
                
                job["status"] = "COMPLETED"
                job["output_file"] = raw_file_name
                updated = True
            elif state == 'JOB_STATE_FAILED':
                job["status"] = "FAILED"
                updated = True
        except Exception as e:
            print(f"  Error: {e}")

    if updated:
        with open("gemini_jobs.jsonl", "w") as f:
            for j in jobs: f.write(json.dumps(j) + "\n")

def parse_and_save_content(content_bytes, model_name, output_dir):
    """Parse the downloaded byte stream and save into corresponding task files"""
    lines = content_bytes.decode('utf-8').splitlines()
    task_files = {}
    model_short = model_name.replace("/", "-")

    for line in lines:
        if not line.strip(): continue
        item = json.loads(line)
        parts = item.get("key", "").split("|")
        if len(parts) < 3: continue
        
        task_name, filename = parts[0], parts[2]
        try:
            resp = item["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except: resp = "[ERROR]"

        record = {"audio_file": filename, "model": model_short, "task": task_name, "response": resp.strip()}
        out_path = os.path.join(output_dir, f"{task_name}_{model_short}_generations.jsonl")
        
        if out_path not in task_files: task_files[out_path] = []
        task_files[out_path].append(record)

    for path, records in task_files.items():
        with open(path, "a", encoding="utf-8") as f:
            for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ================= Main Program =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=True, choices=["submit", "retrieve", "parse"])
    parser.add_argument("-d", "--dimension", type=str, default="gender")
    parser.add_argument("-e", "--exp_id", type=str, default=None)
    parser.add_argument("-t", "--task", type=str, default="all")
    parser.add_argument("--parse_file", type=str, help="Path to the results.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY."); return

    # Initialize new Client
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    exp_id = args.exp_id or f"gemini_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(RESULTS_ROOT, exp_id, args.dimension)
    ensure_dir(exp_dir)

    if args.task == "all":
        tasks_to_run = TASKS
    else:
        tasks_to_run = {args.task: TASKS[args.task]} if args.task in TASKS else {}

    if args.mode == "submit":
        audio_files = get_audio_paths(args.dimension)
        file_uris = upload_audio_files(audio_files, client)
        create_batch_job(MODEL_NAME, tasks_to_run, audio_files, file_uris, args.dimension, client)

    elif args.mode == "retrieve":
        retrieve_and_auto_parse(client, exp_dir)

    elif args.mode == "parse":
        if not args.parse_file: print("Need --parse_file"); return
        with open(args.parse_file, "rb") as f: content = f.read()
        parse_and_save_content(content, MODEL_NAME, exp_dir)

if __name__ == "__main__":
    main()