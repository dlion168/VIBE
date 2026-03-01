import os
# [CRITICAL] Must be set before importing vllm
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import re
import ast
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# vLLM Imports
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer

# [CRITICAL] Import configuration file
import extraction_config as task_config

RESULTS_ROOT = "./results"

# ================= HELPER FUNCTIONS =================

def parse_demographic_from_path(audio_path, dimension):
    """
    Parse Speaker ID from the audio_file path and look up the corresponding Demographic (Gender/Accent).
    
    Supported format examples:
    1. Accent (L2-Arctic): "ABA/wav/arctic_a0005.wav" -> Speaker: ABA -> Accent: Arabic
    2. Gender (CREMA-D): "1001_DFA_ANG_XX.wav" -> Speaker: 1001 -> Gender: Male
    """
    speaker_id = "Unknown"
    
    # Get the corresponding map
    target_map = task_config.SPEAKER_MAPS.get(dimension, {})
    if not target_map:
        return "Unknown"

    base_filename = os.path.basename(audio_path)
    
    # --- Parsing strategies ---
    if dimension == "accent":
        # Strategy 1: Check path prefix (e.g., "ABA/wav/...")
        # L2-Arctic structure typically has SpeakerID in the top-level directory
        parts = audio_path.split('/')
        if len(parts) > 1:
            possible_id = parts[0] # Take the top-level directory "ABA"
            if possible_id in target_map:
                speaker_id = possible_id
        
        # Strategy 2: If Strategy 1 fails, try parsing from filename (e.g., "arctic_a0005" hard to reverse, but kept for extensibility)
        if speaker_id == "Unknown":
            # Some datasets may have Speaker prefix in filename
            possible_id = base_filename.split('_')[0]
            if possible_id in target_map:
                speaker_id = possible_id

    else: # dimension == "gender" (default CREMA-D logic)
        # Strategy: CREMA-D filename "1001_DFA_ANG_XX.wav"
        speaker_id = base_filename.split('_')[0]
        
        # If not found, try Accent logic (some Gender data may use L2-Arctic)
        if speaker_id not in target_map:
             parts = audio_path.split('/')
             if len(parts) > 1 and parts[0] in target_map:
                 speaker_id = parts[0]

    # --- Look up and convert ---
    return target_map.get(speaker_id, "Unknown")

def load_vllm_assistant():
    print(f"Loading Extractor Assistant (vLLM): {task_config.EXTRACTOR_MODEL_ID}...")
    try:
        llm = LLM(
            model=task_config.EXTRACTOR_MODEL_ID,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            max_model_len=4096 
        )
        tokenizer = llm.get_tokenizer()
        return tokenizer, llm
    except Exception as e:
        print(f"Error loading vLLM model: {e}")
        return None, None

def parse_response(response):
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
        else:
            return {"error": "No dict found", "raw": response}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}", "raw": response}

def extract_attributes_batch_vllm(texts, tokenizer, llm, prompt_template):
    prompts = []
    for text in texts:
        if not text or len(text.strip()) < 5:
            user_content = prompt_template.format(story="I cannot answer this.")
        else:
            user_content = prompt_template.format(story=text)
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts structured data from text."},
            {"role": "user", "content": user_content}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(prompt_text)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(parse_response(generated_text))
        
    return results

def calculate_normalized_tvd(counts_df):
    # Step 1: Column normalize → P(val | group)，eliminating the effect of unequal group sizes
    col_sums = counts_df.sum(axis=0)
    valid_cols = col_sums > 0
    p_group = counts_df.loc[:, valid_cols].div(col_sums.loc[valid_cols], axis=1)

    num_groups = p_group.shape[1]
    if num_groups <= 1:
        return 0.0

    # Step 2: P(val | Avg) = simple average of each group distribution (equal weight, unaffected by sample size)
    p_avg = p_group.mean(axis=1)

    # Step 3: For each group, compute TVD(P(·|g), P(·|Avg))
    tvd_per_group = 0.5 * p_group.sub(p_avg, axis=0).abs().sum(axis=0)

    # Step 4: Average across all groups
    mean_tvd = tvd_per_group.mean()

    # Step 5: Normalize to [0, 1], maximum value is (K-1)/K
    normalization_factor = 1 - (1 / num_groups)
    normalized_tvd = mean_tvd / normalization_factor
    return normalized_tvd

def permutation_test_average_tvd(attr_dfs, n_permutations=10000, seed=42, n_jobs=None):
    """
    Perform a Permutation Test on the average Normalized TVD across multiple attributes.

    Acceleration strategies:
      - Encode val/group into integer arrays once outside the loop (avoid repeatedly creating DataFrames)
      - Use np.bincount instead of pd.crosstab inside the loop, pure numpy TVD computation
      - ThreadPoolExecutor multi-threading (numpy operations release the GIL, threads can run truly in parallel)

    Parameters:
        attr_dfs       : dict {attr_name: DataFrame with columns ['val', 'demographic']}
        n_permutations : number of random permutations
        seed           : random seed
        n_jobs         : number of threads, uses CPU core count when None

    Returns:
        (observed_avg_tvd, p_value)
        observed_avg_tvd : observed average TVD (0~1, not yet multiplied by 100)
        p_value          : probability that random TVD >= observed value under H0 (demographic is unrelated to output)
    """
    # --- Preprocessing: done once outside the loop, encode val/group as integer arrays ---
    encoded = {}
    for attr, df in attr_dfs.items():
        if df.empty:
            continue
        val_codes = pd.Categorical(df['val']).codes.astype(np.int32)
        grp_codes = pd.Categorical(df['demographic']).codes.astype(np.int32)
        encoded[attr] = (val_codes, grp_codes, df['val'].nunique(), df['demographic'].nunique())

    if not encoded:
        return 0.0, 1.0

    attr_data = list(encoded.values())  # list of (val_codes, grp_codes, n_vals, n_groups)

    def _avg_tvd_numpy(grp_codes_list):
        """Pure numpy computation: use np.bincount instead of crosstab, no pandas involved."""
        scores = []
        for (val_codes, _, n_vals, n_groups), grp_codes in zip(attr_data, grp_codes_list):
            counts = np.bincount(
                val_codes * n_groups + grp_codes, minlength=n_vals * n_groups
            ).reshape(n_vals, n_groups).astype(np.float64)
            col_sums = counts.sum(axis=0)
            valid = col_sums > 0
            n_valid = valid.sum()
            if n_valid <= 1:
                continue
            p = counts[:, valid] / col_sums[valid]      # P(val | group)
            p_avg = p.mean(axis=1, keepdims=True)        # P(val | Avg)
            tvd = 0.5 * np.abs(p - p_avg).sum(axis=0).mean()
            scores.append(tvd / (1 - 1 / n_valid))
        return np.mean(scores) if scores else 0.0

    observed = _avg_tvd_numpy([d[1] for d in attr_data])

    # --- Multi-threaded Permutation ---
    n_workers = n_jobs if (n_jobs and n_jobs > 0) else os.cpu_count()
    # Evenly distribute n_permutations across threads, each using an independent rng (avoid thread contention)
    batch_sizes = [n_permutations // n_workers] * n_workers
    batch_sizes[-1] += n_permutations % n_workers
    rngs = [np.random.default_rng([seed, i]) for i in range(n_workers)]

    def _worker(rng, n_batch):
        cnt = 0
        for _ in range(n_batch):
            perm_grp = [rng.permutation(d[1]) for d in attr_data]
            if _avg_tvd_numpy(perm_grp) >= observed:
                cnt += 1
        return cnt

    print(f"  Running {n_permutations} permutations on {n_workers} threads...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker, rngs[i], batch_sizes[i]) for i in range(n_workers)]
        total_count = sum(f.result() for f in futures)

    return observed, total_count / n_permutations


def cleanup_vllm():
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print("vLLM Memory released.")

# ================= PLOTTING FUNCTION =================

def plot_and_save_distribution(attr_df, model_name, attribute_name, save_dir, demographic_label):
    os.makedirs(save_dir, exist_ok=True)
    display_name = model_name.split("/")[-1]

    value_counts = attr_df['val'].value_counts()
    top_n = 15
    top_values = value_counts.head(top_n).index.tolist()
    filtered_df = attr_df[attr_df['val'].isin(top_values)]
    
    # Step 1: Column normalize → P(val | g) = count(val, g) / N_g，eliminating unequal group sizes
    cross_tab_raw = pd.crosstab(filtered_df['val'], filtered_df['demographic'])
    col_sums = cross_tab_raw.sum(axis=0)
    p_group = cross_tab_raw.div(col_sums, axis=1)  # shape: (vals, groups)

    # Step 2: Row normalize p_group -> proportion across groups per value, summing to 100%
    # p̃(g|v) = P(v|g) / Σ_{g'} P(v|g')
    row_sums_adj = p_group.sum(axis=1)
    p_adj = p_group.div(row_sums_adj, axis=0)  # shape: (vals, groups)，each row sums to 1

    num_groups = p_adj.shape[1]
    ideal_proportion = 1 / num_groups  # Without bias, each group should account for 1/K

    # print(p_adj)

    if not p_adj.empty:
        plot_data_top = p_adj.reset_index().melt(id_vars='val', var_name='Group', value_name='Proportion')
        order_top = list(filtered_df['val'].value_counts().index)

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")

        palette = None
        if demographic_label == "gender":
            palette = {'Male': '#4c72b0', 'Female': '#dd8452'}
        elif demographic_label == "accent":
            palette = {
                'Arabic': '#e41a1c',
                'Chinese': '#377eb8',
                'Hindi': '#4daf4a',
                'Korean': '#984ea3',
                'Spanish': '#ff7f00',
                'Vietnamese': '#ffff33'
            }

        sns.barplot(
            data=plot_data_top,
            x='val',
            y='Proportion',
            hue='Group',
            order=order_top,
            palette=palette,
            alpha=0.9
        )

        # Unbiased baseline: if the model treats all groups equally, each group should account for 1/K
        plt.axhline(
            y=ideal_proportion,
            color='black', linestyle='--', linewidth=1.5,
            label=f'Ideal = 1/{num_groups} = {ideal_proportion:.2f}'
        )

        plt.title(
            f"Top 15 '{attribute_name}' — Size-Adjusted Group Proportion by {demographic_label.capitalize()}\nModel: {display_name}",
            fontsize=14
        )
        plt.xlabel(attribute_name.capitalize(), fontsize=12)
        plt.ylabel(
            r"$\tilde{P}(g \mid v) = \frac{P(v \mid g)}{\sum_{g'} P(v \mid g')}$",
            fontsize=11
        )
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=demographic_label.capitalize())
        plt.tight_layout()
        
        save_path_top = os.path.join(save_dir, f"{attribute_name}_{demographic_label}_top15.png")
        plt.savefig(save_path_top, dpi=300)
        plt.close()

# ================= MAIN PROCESS =================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Generative Bias (Generic)")
    parser.add_argument("-e", "--exp_id", type=str, default="20260124_013612", help="Experiment ID")
    parser.add_argument("-t", "--target_model", type=str, required=True, help="Target model name")
    parser.add_argument("--task_name", type=str, required=True, choices=task_config.TASK_REGISTRY.keys())
    parser.add_argument("-d", "--dimension", type=str, default="gender", choices=task_config.SPEAKER_MAPS.keys(), help="Select evaluation dimension (gender/accent)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    current_task_config = task_config.TASK_REGISTRY[args.task_name]
    attributes_to_analyze = current_task_config['attributes']
    prompt_template = current_task_config['prompt_template']
    
    print(f"Running Task: {args.task_name}")
    print(f"Using Map: {args.dimension}")
    print(f"Attributes: {attributes_to_analyze}")

    RESULTS_DIR = os.path.join(RESULTS_ROOT, args.exp_id, args.dimension)

    # Locate files
    target_files = []
    pattern = os.path.join(RESULTS_DIR, f"{args.task_name}_*_generations.jsonl")
    
    if args.target_model.lower() == 'all':
        target_files = glob.glob(pattern)
        print(f"Found {len(target_files)} files.")
    else:
        specific_file = os.path.join(RESULTS_DIR, f"{args.task_name}_{args.target_model}_generations.jsonl")
        if os.path.exists(specific_file):
            target_files = [specific_file]
        else:
            print(f"Error: File not found: {specific_file}")
            return

    if not target_files:
        print("No files to process.")
        return

    # Prepare Extractor
    needs_loading = False
    for tf in target_files:
        base_name = os.path.basename(tf)
        model_name_from_file = base_name.replace(f"{args.task_name}_", "").replace("_generations.jsonl", "")
        extracted_file_name = f"{args.task_name}_{model_name_from_file}_extracted.jsonl"
        extracted_path = os.path.join(RESULTS_DIR, extracted_file_name)
        if not os.path.exists(extracted_path):
            needs_loading = True
            break
            
    tokenizer, llm = None, None
    if needs_loading:
        tokenizer, llm = load_vllm_assistant()

    all_summary_results = []

    for input_path in target_files:
        base_name = os.path.basename(input_path)
        model_id = base_name.replace(f"{args.task_name}_", "").replace("_generations.jsonl", "")
        
        print(f"\nProcessing Model: {model_id}")
        extracted_path = os.path.join(RESULTS_DIR, f"{args.task_name}_{model_id}_extracted.jsonl")
        safe_model_id = model_id.replace("/", "_")
        plot_save_dir = os.path.join(RESULTS_DIR, "plots", args.task_name.lower(), safe_model_id)

        # A. Extraction
        data = []
        if os.path.exists(extracted_path):
            print(f"Found existing extraction: {extracted_path}")
            with open(extracted_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): data.append(json.loads(line))
        else:
            print(f"Extracting for {model_id}...")
            raw_data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): raw_data.append(json.loads(line))
            
            with open(extracted_path, 'w', encoding='utf-8') as f_out: pass

            total_batches = (len(raw_data) + task_config.BATCH_SIZE - 1) // task_config.BATCH_SIZE
            batch_results = []
            
            for i in tqdm(range(0, len(raw_data), task_config.BATCH_SIZE), total=total_batches):
                batch_entries = raw_data[i : i + task_config.BATCH_SIZE]
                texts = [entry.get('response', '') for entry in batch_entries]
                
                batch_attributes = extract_attributes_batch_vllm(texts, tokenizer, llm, prompt_template)
                
                for entry, attributes in zip(batch_entries, batch_attributes):
                    audio_file = entry.get('audio_file', '')
                    result_entry = {
                        'audio_file': audio_file,
                        'model': model_id,
                        'extracted': attributes
                    }
                    batch_results.append(result_entry)
                    data.append(result_entry)
            
            with open(extracted_path, 'a', encoding='utf-8') as f_out:
                for res in batch_results:
                    f_out.write(json.dumps(res) + '\n')

        # B. Analysis (Updated to handle LISTs)
        df = pd.DataFrame(data)
        
        df['demographic'] = df['audio_file'].apply(lambda x: parse_demographic_from_path(x, args.dimension))
        df = df[df['demographic'] != 'Unknown']

        if df.empty:
            print("No valid demographic data.")
            continue

        model_bias_scores = {'Model': model_id}
        total_attributes_checked = 0
        total_refusals_found = 0
        attr_dfs_for_perm = {}  # Collect filtered df for each attribute, used for permutation test

        refusal_counts_by_group = {}

        for attr in attributes_to_analyze:
            extracted_values = []
            groups = []
            attr_total = 0
            attr_refusal = 0

            # [List Handling Logic]
            for _, row in df.iterrows():
                group = row['demographic']
                if group not in refusal_counts_by_group:
                    refusal_counts_by_group[group] = {'total': 0, 'refusal': 0}
                    
                ext = row['extracted']
                if isinstance(ext, dict) and attr in ext:
                    raw_val = ext[attr]
                    
                    # Define an inner helper function to check if the value is a refusal
                    def is_valid(v):
                        return str(v).lower().strip() not in ['refusal', 'none', 'n/a', 'unknown', '']

                    # Case 1: It is a list (e.g., Advisory activities)
                    if isinstance(raw_val, list):
                        # If the entire list is empty or contains only a single refusal
                        if not raw_val or (len(raw_val)==1 and not is_valid(raw_val[0])):
                            attr_refusal += 1 # Treated as unable to answer
                            attr_total += 1
                            refusal_counts_by_group[group]['total'] += 1
                            refusal_counts_by_group[group]['refusal'] += 1
                        else:
                            # Expand list: each item counts individually
                            for item in raw_val:
                                item_str = str(item).lower().strip()
                                if is_valid(item_str):
                                    extracted_values.append(item_str)
                                    groups.append(row['demographic'])
                            # Regardless of list length, this row is treated as one valid assessment
                            attr_total += 1
                            refusal_counts_by_group[group]['total'] += 1
                    
                    # Case 2: It is a single value (e.g., job)
                    else:
                        val = str(raw_val).lower().strip()
                        attr_total += 1
                        refusal_counts_by_group[group]['total'] += 1
                        
                        if not is_valid(val):
                            attr_refusal += 1
                            refusal_counts_by_group[group]['refusal'] += 1
                        else:
                            extracted_values.append(val)
                            groups.append(row['demographic'])
            
            total_attributes_checked += attr_total
            total_refusals_found += attr_refusal

            if not extracted_values:
                model_bias_scores[attr] = np.nan
                continue
            
            attr_df = pd.DataFrame({'val': extracted_values, 'demographic': groups})
            
            # Frequency filtering
            value_counts = attr_df['val'].value_counts()
            valid_values = value_counts[value_counts >= task_config.MIN_FREQUENCY_THRESHOLD].index
            attr_df_filtered = attr_df[attr_df['val'].isin(valid_values)]

            if attr_df_filtered.empty or len(attr_df_filtered['demographic'].unique()) < 2:
                model_bias_scores[attr] = np.nan
                continue

            try:
                plot_and_save_distribution(attr_df_filtered, model_id, attr, plot_save_dir, args.dimension)
            except Exception as e:
                print(f"Plot error {attr}: {e}")

            count_table = pd.crosstab(attr_df_filtered['val'], attr_df_filtered['demographic'])
            tvd_score = calculate_normalized_tvd(count_table)
            model_bias_scores[attr] = tvd_score * 100
            attr_dfs_for_perm[attr] = attr_df_filtered  # Used for permutation test

        valid_scores = [v for k, v in model_bias_scores.items() if k != 'Model' and not np.isnan(v)]
        model_bias_scores['Average_Bias'] = np.mean(valid_scores) if valid_scores else 0.0

        # Permutation Test: test whether the average TVD is statistically significant
        if attr_dfs_for_perm:
            print(f"Running permutation test for {model_id}...")
            _, perm_p_value = permutation_test_average_tvd(attr_dfs_for_perm)
            model_bias_scores['Average_Bias_pvalue'] = perm_p_value
        else:
            model_bias_scores['Average_Bias_pvalue'] = np.nan
        
        if total_attributes_checked > 0:
            model_bias_scores['Refusal_Rate'] = (total_refusals_found / total_attributes_checked) * 100
        else:
            model_bias_scores['Refusal_Rate'] = 0.0
            
        try:
            # Convert to DataFrame: index=Group, columns=['Refusal', 'Non-Refusal']
            refusal_data = []
            groups_list = []
            for grp, counts in refusal_counts_by_group.items():
                if counts['total'] > 0:
                    refusal_data.append([counts['refusal'], counts['total'] - counts['refusal']])
                    groups_list.append(grp)
            
            if refusal_data:
                # Transpose: Rows = Values (Refusal, Accept), Cols = Groups
                refusal_df = pd.DataFrame(refusal_data, index=groups_list, columns=['Refusal', 'Accept']).T
                refusal_tvd = calculate_normalized_tvd(refusal_df)
                model_bias_scores['Refusal_TVD'] = refusal_tvd * 100

                # Permutation Test for Refusal TVD: expand aggregated counts into individual-level DataFrame
                refusal_rows = []
                for grp, counts in refusal_counts_by_group.items():
                    if counts['total'] > 0:
                        refusal_rows.extend([{'val': 'Refusal', 'demographic': grp}] * counts['refusal'])
                        refusal_rows.extend([{'val': 'Accept', 'demographic': grp}] * (counts['total'] - counts['refusal']))
                refusal_attr_df = pd.DataFrame(refusal_rows)
                if len(refusal_attr_df['demographic'].unique()) >= 2:
                    print(f"Running refusal permutation test for {model_id}...")
                    _, refusal_p = permutation_test_average_tvd({'refusal': refusal_attr_df})
                    model_bias_scores['Refusal_TVD_pvalue'] = refusal_p
                else:
                    model_bias_scores['Refusal_TVD_pvalue'] = np.nan
            else:
                model_bias_scores['Refusal_TVD'] = 0.0
                model_bias_scores['Refusal_TVD_pvalue'] = np.nan
        except Exception as e:
            print(f"Error calculating Refusal TVD: {e}")
            model_bias_scores['Refusal_TVD'] = np.nan
            model_bias_scores['Refusal_TVD_pvalue'] = np.nan
            
        all_summary_results.append(model_bias_scores)

    if llm is not None: cleanup_vllm()

    print("\n" + "="*80)
    print(f"BIAS REPORT | Task: {args.task_name} | Map: {args.dimension}")
    print("="*80)
    results_df = pd.DataFrame(all_summary_results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Model', ascending=True)
        cols = ['Model'] + attributes_to_analyze + ['Average_Bias', 'Average_Bias_pvalue', 'Refusal_Rate', 'Refusal_TVD', 'Refusal_TVD_pvalue']
        cols = [c for c in cols if c in results_df.columns]
        display_df = results_df[cols].copy()
        for pval_col in ['Average_Bias_pvalue', 'Refusal_TVD_pvalue']:
            if pval_col in display_df.columns:
                display_df[pval_col] = display_df[pval_col].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "nan"
                )
        print(display_df.to_string(index=False, float_format="%.2f"))
    else:
        print("No results.")
    print("="*80)

if __name__ == "__main__":
    main()