import json
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import os

TRUTH_PATH = Path("../data/annotations_level_2.json")
RESULTS_STRING = "final_results"
VOTE_THRESHOLD = 3
OUTPUT_JSON = f"../aggregated_results/final_results/F1_{RESULTS_STRING}_threshold_{VOTE_THRESHOLD}_labels.json"
PRED_PATH = f"../aggregated_results/final_results/{RESULTS_STRING}_threshold_{VOTE_THRESHOLD}_labels.json"

def split_labels(s):
    return [t.strip() for t in s.split(",")
            if t.strip() and t.strip().lower()!="none"]

def load_truth():
    truth_map = {}
    with open(TRUTH_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        sent = entry["sentence"]
        for ann, lbl_str in entry.get("values", {}).items():
            truth_map[(sent, ann)] = split_labels(lbl_str)
    return truth_map

def load_preds():
    with open(PRED_PATH, encoding="utf-8") as f:
        return json.load(f)

def evaluate_baseline():
    baseline_json = f"../aggregated_results/final_results/{RESULTS_STRING}_baseline_k0_labels.json"
    output_json = f"../aggregated_results/final_results/F1_{RESULTS_STRING}_k0_baseline.json"

    truth = load_truth()

    with open(baseline_json, encoding="utf-8") as f:
        preds = json.load(f)

    all_labels = set()
    for labs in truth.values():
        all_labels.update(labs)
    for sent, annots in preds.items():
        for ann, taskdict in annots.items():
            for label_str in taskdict.values():
                if isinstance(label_str, str):
                    all_labels.update(split_labels(label_str))

    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    mlb.fit([])

    out = {}
    for sent, annots in preds.items():
        out[sent] = {}
        for ann, taskdict in annots.items():
            true_vals = truth.get((sent, ann), [])

            pred_vals = []
            for label_str in taskdict.values():
                if isinstance(label_str, str):
                    pred_vals.extend(split_labels(label_str))

            pred_vals = sorted(set(pred_vals))

            if not true_vals and not pred_vals:
                score = 1.0
            elif (not true_vals and pred_vals) or (true_vals and not pred_vals):
                score = 0.0
            else:
                y_true = mlb.transform([true_vals])
                y_pred = mlb.transform([pred_vals])
                score = f1_score(y_true, y_pred, average="micro", zero_division=0)

            out[sent][ann] = round(score, 4)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Wrote per-sentence baseline F1 scores to", output_json)

def compute_breakdown_results_baseline():
    INPUT_JSON = "../aggregated_results/final_results/F1_final_results_k0_baseline.json"
    OUTPUT_BREAKDOWN_JSON = "../aggregated_results/final_results/break_down_results_baseline.json"

    with open(INPUT_JSON, 'r', encoding="utf-8") as f:
        data = json.load(f)

    annotators = set()
    for scores in data.values():
        annotators.update(scores.keys())
    annotators = sorted(annotators)

    # Compute average score per annotator
    averages = {}
    for ann in annotators:
        scores = [scores.get(ann, 0) for scores in data.values()]
        averages[ann] = sum(scores) / len(scores)

    print("Average scores per annotator:")
    for ann, avg in averages.items():
        print(f"{ann}: {avg:.4f}")

    final_results = {"average_scores": averages}
    with open(OUTPUT_BREAKDOWN_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("Wrote breakdown results to", OUTPUT_BREAKDOWN_JSON)

def breakdown_per_annotator_k():
    input_path = "../aggregated_results/final_results/F1_final_results_threshold_3_labels.json"
    output_json = "../aggregated_results/final_results/break_down_results.json"
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for statement, annos in data.items():
        for annotator, kdict in annos.items():
            for k, metrics in kdict.items():
                for metric, value in metrics.items():
                    records.append({
                        "annotator": annotator,
                        "k": int(k.split('=')[1]),
                        "metric": metric,
                        "value": value
                    })

    df = pd.DataFrame(records)
    result = df.groupby(["annotator", "k", "metric"])["value"] \
               .mean() \
               .reset_index() \
               .pivot(index=["annotator", "k"], columns="metric", values="value") \
               .reset_index()

    print(result)

    if output_json:
        out = {}
        for _, row in result.iterrows():
            ann = row['annotator']
            k = f"k={int(row['k'])}"
            out.setdefault(ann, {})[k] = {
                m: row[m]
                for m in ["sentiment","emotion","argument","topic","all"]
                if m in row
            }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote breakdown to {output_json}")


def main():
    truth = load_truth()
    preds = load_preds()

    # collect vocabulary
    all_labels = set()
    for labs in truth.values():
        all_labels.update(labs)
    for sent, annots in preds.items():
        for ann, kdict in annots.items():
            for k, tasks in kdict.items():
                for task in tasks.values():
                    all_labels.update(split_labels(task))

    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    mlb.fit([])

    out = {}
    for sent, annots in preds.items():
        out[sent] = {}
        for ann, kdict in annots.items():
            out[sent][ann] = {}
            true_vals = truth.get((sent, ann), [])
            for k, tasks in kdict.items():
                out[sent][ann][k] = {}
                for task in ["argument", "emotion", "sentiment", "topic", "all"]:
                    pred_vals = split_labels(tasks.get(task, ""))

                    # special-case both empty → score=1.0; one empty → 0.0
                    if not true_vals and not pred_vals:
                        score = 1.0
                    elif (not true_vals and pred_vals) or (true_vals and not pred_vals):
                        score = 0.0
                    else:
                        y_true = mlb.transform([true_vals])
                        y_pred = mlb.transform([pred_vals])
                        # zero_division=0 avoids warnings on labels missing
                        score = f1_score(y_true, y_pred, average="micro",
                                         zero_division=0)

                    out[sent][ann][k][task] = round(score, 4)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Wrote per-sentence values F1 to", OUTPUT_JSON)


def plot_avg_cohen_kappa_by_annotator():
    """
    For each k in [0,1,5,10,15]:
      - For each of the 5 human annotators:
          * Build a binary vector of presence/absence over all labels for
            (sentence, annotator) comparing human vs. model at that k.
          * Compute Cohen’s κ on that vector.
      - Average the 5 annotators.
    Then plot Avg κ vs. k.
    """
    truth = load_truth()
    preds = load_preds()
    # load zero-shot baseline per annotator
    baseline_file = Path(
      f"../aggregated_results/final_results/"
      f"{RESULTS_STRING}_baseline_k0_labels.json"
    )
    with open(baseline_file, encoding="utf-8") as f:
        zeroshot = json.load(f)

    # build the full set of labels
    all_labels = set(truth.values().__iter__().__next__())
    for labs in truth.values():
        all_labels.update(labs)
    for sent, annots in preds.items():
        for shot_dicts in annots.values():
            for shot in shot_dicts.values():
                all_labels.update(split_labels(shot.get("all","")))
    for sent, annots in zeroshot.items():
        for shot in annots.values():
            if isinstance(shot, dict):
                for txt in shot.values():
                    all_labels.update(split_labels(txt))
            else:
                all_labels.update(split_labels(shot))
    all_labels = sorted(all_labels)

    # identify the 5 annotator IDs
    annotators = sorted({ann for (_,ann) in truth.keys()})
    ks = [0, 1, 5, 10, 15]
    avg_kappas = []

    for k in ks:
        kappas = []
        for ann in annotators:
            y_true = []
            y_pred = []
            # collect every sentence this annotator labeled
            for (sent, a2), true_labs in truth.items():
                if a2 != ann:
                    continue
                # get the model's prediction string for this annotator
                if k == 0:
                    shot = zeroshot[sent][ann]
                    pred_str = shot.get("all","") if isinstance(shot, dict) else shot
                else:
                    pred_str = preds[sent][ann].get(f"k={k}", {}).get("all","")

                pred_labs = split_labels(pred_str)
                for lab in all_labels:
                    y_true.append(int(lab in true_labs))
                    y_pred.append(int(lab in pred_labs))

            # compute κ for this annotator, this k
            if y_true:
                kappas.append(cohen_kappa_score(y_true, y_pred))

        # average across annotators
        avg_kappas.append(np.mean(kappas) if kappas else 0.0)

    plt.figure(figsize=(6,4))
    plt.plot(ks, avg_kappas, marker='o', linewidth=2)
    plt.xticks(ks)
    plt.xlabel("Number of examples (k)")
    plt.ylabel("Average Cohen’s κ\n(across 5 annotators)")
    plt.title("Model–Human Agreement")
    plt.grid(True)
    plt.tight_layout()

    out_path = Path("../aggregated_results/final_results") / "avg_cohen_kappa_per_annotator.png"
    plt.savefig(out_path)
    print("Saved average Cohen’s κ plot to", out_path)


if __name__ == "__main__":
    main()
    evaluate_baseline()
    compute_breakdown_results_baseline()
    breakdown_per_annotator_k()
    plot_avg_cohen_kappa_by_annotator()
