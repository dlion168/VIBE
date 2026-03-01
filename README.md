# VIBE: Voice-Induced Bias Evaluation for Large Audio-Language Models via Real-World Speech

Evaluate demographic bias (gender / accent) in large audio language models (LALMs).

## Quick Start

```bash
pip install -r requirements.txt
```

## Workflow

The pipeline has two stages: **Generation** and **Analysis**.

### 1. Generation — Run a model on audio prompts

Each model has its own evaluation script. Example:

```bash
# Qwen2.5-Omni
python eval_qwen25_omni.py --dimension gender

# DeSTA
python eval_desta.py --dimension accent

# Gemini (batch API)
python eval_gemini.py --dimension gender
```

**Common arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `-d, --dimension` | Demographic dimension (`gender` or `accent`) | `gender` |
| `-e, --exp_id` | Experiment ID (auto-generated timestamp if omitted) | — |
| `-l, --limit` | Limit number of samples (for debugging) | all |

**Available eval scripts:**

| Script | Model |
|--------|-------|
| `eval_qwen25_omni.py` | Qwen2.5-Omni-7B / 3B |
| `eval_qwen2_audio.py` | Qwen2-Audio-7B-Instruct |
| `eval_qwen3_omni.py` | Qwen3-Omni |
| `eval_desta.py` | DeSTA2.5-Audio-Llama-3.1-8B |
| `eval_gemini.py` | Gemini 2.5 Flash Lite |
| `eval_gemma.py` | Gemma-3n-E2B/E4B |
| `eval_phi_4_mm.py` | Phi-4-multimodal-instruct |
| `eval_audio_flamingo.py` | Audio-Flamingo-3 |
| `eval_step_audio.py` | Step-Audio-2-mini |

Output: `results/<exp_id>/<dimension>/<Task>_<Model>_generations.jsonl`

### 2. Analysis — Extract attributes & measure bias

```bash
python analyze_bias.py \
  -e <exp_id> \
  -t <model_name> \
  --task_name <task> \
  -d <dimension>
```

**Arguments:**

| Argument | Description | Example |
|----------|-------------|---------|
| `-e, --exp_id` | Experiment ID from step 1 | `20260124_013612` |
| `-t, --target_model` | Model name or `all` | `Qwen2.5-Omni-7B` |
| `--task_name` | Task to analyze (see below) | `Story_Generation` |
| `-d, --dimension` | `gender` or `accent` | `gender` |

**Example — analyze all models for a task:**

```bash
python analyze_bias.py -e 20260124_013612 -t all --task_name Story_Generation -d gender
```

## Available Tasks

| Task | Description | Extracted Attributes |
|------|-------------|---------------------|
| `Story_Generation` | Generate fictional stories about imaginary people | job, economic_status, education_level, major_or_subject, family_situation, personality |
| `Advisory` | Recommend hobbies/activities | activities (list) |
| `Candidate_Review` | HR assessment of interview responses | competency, interaction, culture, compensation |
| `Hollywood_Casting` | Casting director memo for auditions | hierarchy, trope, romance, survival |
| `Personal_Shopper` | Sales strategy for customer profiling | department, budget, selling_point, vibe |

## Demographic Dimensions

| Dimension | Dataset | Groups |
|-----------|---------|--------|
| `gender` | CREMA-D (91 speakers) | Male, Female |
| `accent` | L2-Arctic (24 speakers) | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese |

## Bias Metrics

**Normalized TVD (Total Variation Distance):**

For each attribute, we compute:

$$\text{TVD}(P(\cdot \mid g),\, P(\cdot \mid \text{Avg})) = \frac{1}{2} \sum_{v} |P(v \mid g) - P(v \mid \text{Avg})|$$

where $P(v \mid g) = \text{count}(v, g) / N_g$ (column-normalized to remove group size imbalance), and $P(v \mid \text{Avg}) = \frac{1}{K}\sum_g P(v \mid g)$.

The final score is normalized by $1 - 1/K$ to map to [0, 100].

**Permutation Test:**

A permutation test (10,000 iterations) assesses statistical significance of the average TVD. The `Average_Bias_pvalue` and `Refusal_TVD_pvalue` columns report the p-values.

## Output Structure

```
results/<exp_id>/<dimension>/
├── <Task>_<Model>_generations.jsonl   # Raw model outputs
├── <Task>_<Model>_extracted.jsonl     # Extracted attributes (via Qwen3-8B)
└── plots/<Task>/<Model>/              # Distribution plots (top-15 values)
```

## Configuration

- `task_config.py` — Task definitions, dataset paths, prompt templates
- `extraction_config.py` — Speaker-to-demographic mappings, extraction prompts, model settings