# Annotating Values with Language Models

This repository aims to automatically annotate sentences with human values based on previous annotations of other subjective tasks (argument, topic, emotion, and sentiment). It supports both task-specific and multi-task setups (detecting value based on only one task or all), using a K-nearest neighbor retrieval strategy for selecting examples for the few-shot prompt.

## Method Overview

Given a sentence and its existing annotations for tasks such as emotion, sentiment, topic, and argumentation, the system prompts a language model to predict a list of values. The few-shot examples used in the prompt are retrieved using sentence embeddings and KNN search. Two prompting modes are supported:

- **Per-task (`--mode per`)**: Prompts are specific to each task.
- **All-task (`--mode all`)**: A combined prompt uses annotations from all tasks at once.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/adina-dobrinoiu/Annotating_values.git
cd Annotating_values
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the main annotation script with:

```bash
python -m scripts.run --k 3 --mode per --seed 42
```

- `--k`: Number of few-shot examples (0 = zero-shot, 1 = one-shot, 2+ = few-shot). Note that when using k = 1, only the annotations of the current sentence will be included. To give examples from similar sentences use k > 1.
- `--mode`: Prompting mode: `per` for task-specific (predicting values based only on one task - this will run for all tasks) or `all` for all tasks (predicting values based only on all tasks)
- `--seed`: Random seed for reproducibility

The script `run_batches.sh` can also be used to run multiple experiments.

## Output

Results will be saved to the `results/` directory as JSON files, one per annotator and task, named like:

```
results_argument_Annotator_1_k3_per_seed42.json
results_all_Annotator_3_k2_all_seed123.json
```

Each result entry includes:
- The input sentence
- The generated value labels
- The few-shot examples used (if applicable)

## Processing results
After the results are finish, the following file should be run in this order:
1. `aggregate_results`: aggregate the runs (for different seeds) based on majority vote with threshold `VOTE_THRESHOLD`
2. `evaluate`:  compute the micro average F1 score for the aggregated results
3. `plot`:  plots the results based on different groupings (k value, annotator, task)

The `script` folder also includes the following files:
- `annotator_agreement.py`
- `measure_change`:  measures the changes in prediction, measured as the relative symmetric difference between baseline and alternative approach

# Configuring the Model

The model can be loaded from either a local path or Hugging Face Hub.

### Example `config.yaml`

```yaml
use_local_model: false
hf_token: "your_huggingface_access_token"

hf_model_folder: "/home/youruser/hf-models/"
base_model_name: "meta-llama/Meta-Llama-3-8B-Instruct"

bitsandbytes:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: false
```

### Hugging Face Hub Access

If `use_local_model: false`, ensure you have a valid token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

You may:

- Set the token directly in `config.yaml`
- Or store it in an environment variable:

```bash
export HF_TOKEN=your_token_here
```

Then in your code, use:

```python
import os
hf_token = os.getenv('HF_TOKEN')
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
