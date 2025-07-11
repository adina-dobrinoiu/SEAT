import os
import json
import re
from glob import glob
from collections import defaultdict, Counter

# --- Configuration ---
RESULTS_STRING = "final_results"
RESULTS_DIR = f"../{RESULTS_STRING}"
VOTE_THRESHOLD = 3  # at least 3 out of 5 votes
OUTPUT_FILE = f"../aggregated_results/final_results/{RESULTS_STRING}_threshold_{VOTE_THRESHOLD}_labels.json"
SEEDS = {"seed0", "seed1", "seed2", "seed3", "seed4"}


def extract_metadata(fname):
    """
    Parse filenames of form:
      results_<task>_Annotator_<n>_k<k>_<mode>_seed<seed>.json
    Returns (task, annotator, k, seed) or (None, ...).
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts) != 7 or parts[0] != "results":
        return None, None, None, None
    task = parts[1]
    if parts[2] != "Annotator":
        return None, None, None, None
    annotator = f"{parts[2]}_{parts[3]}"
    if not parts[4].startswith("k"):
        return None, None, None, None
    k = parts[4][1:]
    seed = parts[6]
    if seed not in SEEDS:
        return None, None, None, None
    return task, annotator, k, seed


def extract_first_list(text):
    """
    Find the first '[' ... ']', grab its contents, split on commas,
    strip quotes/whitespace, return list of strings.
    """
    m = re.search(r"\[(.*?)\]", text or "", re.DOTALL)
    if not m:
        return []
    inside = m.group(1)
    items = []
    for part in inside.split(","):
        v = part.strip().strip('"').strip("'")
        if v:
            items.append(v)
    return items


def vote(lists_of_labels, threshold):
    """
    Given list of label‐lists from 5 seeds, returns
    all labels appearing in at least threshold of them,
    joined by ', ', or "" if none.
    """
    flat = [lbl for labels in lists_of_labels for lbl in labels]
    if not flat:
        return ""
    cnt = Counter(flat)
    result = [lbl for lbl, c in cnt.items() if c >= threshold]
    return ", ".join(sorted(result)) if result else ""

def extract_baseline_metadata(fname):
    """
    Extracts (annotator, k, seed) from filenames like:
      results_Annotator_1_k0_seed0.json
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts) != 5 or parts[0] != "results" or parts[1] != "Annotator":
        return None, None, None
    annotator = f"{parts[1]}_{parts[2]}"
    k = parts[3][1:]  # remove 'k'
    seed = parts[4]
    if seed not in SEEDS:
        return None, None, None
    return annotator, k, seed

def generate_k0_baseline_json():
    baseline_groups = defaultdict(list)
    baseline_pattern = os.path.join(RESULTS_DIR, "results_Annotator_*_k0_seed*.json")
    for path in glob(baseline_pattern):
        fname = os.path.basename(path)
        annotator, k, seed = extract_baseline_metadata(fname)
        if annotator and k == "0" and seed:
            baseline_groups[annotator].append(path)

    baseline_output = defaultdict(lambda: defaultdict(dict))

    for annotator, files in baseline_groups.items():
        if len(files) < 5:
            continue

        sentence_to_labels = defaultdict(list)
        for file in files:
            with open(file, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            for entry in data:
                sentence = entry.get("sentence")
                raw = entry.get("generated_response", "")
                labels = extract_first_list(raw)
                sentence_to_labels[sentence].append(labels)

        for sentence, lists in sentence_to_labels.items():
            if len(lists) == 5:
                maj = vote(lists, VOTE_THRESHOLD)
                baseline_output[sentence][annotator]["k0_baseline"] = maj

    baseline_file = f"../aggregated_results/final_results/{RESULTS_STRING}_baseline_k0_labels.json"
    with open(baseline_file, "w", encoding="utf-8") as out:
        json.dump(baseline_output, out, indent=2, ensure_ascii=False)

    print(f"Saved baseline (k=0) labels to {baseline_file}")

def main():
    # Group files by (task, annotator, k)
    groups = defaultdict(list)
    pattern = os.path.join(RESULTS_DIR, "results_*_seed*.json")
    for path in glob(pattern):
        fname = os.path.basename(path)
        task, annotator, k, seed = extract_metadata(fname)
        if task and annotator and k and seed:
            groups[(task, annotator, k)].append(path)

    output = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for (task, annotator, k), files in groups.items():
        # only proceed if we have all 5 seeds
        if len(files) < 5:
            continue

        # collect lists per sentence
        sentence_to_labels = defaultdict(list)
        for file in files:
            with open(file, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            # each file is a list of {sentence, generated_response,...}
            for entry in data:
                sentence = entry.get("sentence")
                raw = entry.get("generated_response", "")
                labels = extract_first_list(raw)
                sentence_to_labels[sentence].append(labels)

        # compute majority vote
        for sentence, lists in sentence_to_labels.items():
            if len(lists) == 5:
                maj = vote(lists, VOTE_THRESHOLD)
                output[sentence][annotator][f"k={k}"][task] = maj

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(output, out, indent=2, ensure_ascii=False)

    print(f"Saved majority-vote JSON to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    generate_k0_baseline_json()
