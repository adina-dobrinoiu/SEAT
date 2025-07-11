import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_STRING = "final_results"
VOTE_THRESHOLD = 3
OUT_DIR = Path("../aggregated_results/final_results/")
AGGREGATED_RESULTS_FILE = OUT_DIR / f"F1_{RESULTS_STRING}_threshold_{VOTE_THRESHOLD}_labels.json"


def load_aggregated_results():
    with open(AGGREGATED_RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_avg_baseline_f1_per_annotator(baseline_path, results_string):
    baseline_file = baseline_path / f"F1_{results_string}_k0_baseline.json"
    if not baseline_file.exists():
        print("Baseline F1 file not found:", baseline_file)
        return {}

    with open(baseline_file, encoding="utf-8") as f:
        baseline_data = json.load(f)

    annotator_scores = {}
    for sent_dict in baseline_data.values():
        for annotator, f1_val in sent_dict.items():
            if isinstance(f1_val, dict):
                f1_val = list(f1_val.values())
            else:
                f1_val = [f1_val]
            annotator_scores.setdefault(annotator, []).extend(f1_val)

    avg_by_annotator = {
        annot: sum(vals) / len(vals)
        for annot, vals in annotator_scores.items()
    }
    return avg_by_annotator


def plot_taskwise_grouped_bars(df, out_path, baseline_json):
    tasks = ["all", "emotion", "sentiment", "topic", "argument"]
    ks = [0, 1, 5, 10, 15]
    labels = ["ZS", "OS", "FS-5", "FS-10", "FS-15"]

    grouped = df[df["task"].isin(tasks)].groupby(["task", "k"])["f1"].mean().unstack("k", fill_value=0)

    with open(baseline_json, encoding="utf-8") as f:
        baseline_data = json.load(f)

    zs_averages = {task: [] for task in tasks}
    for sentence_dict in baseline_data.values():
        for annotator, value in sentence_dict.items():
            if isinstance(value, dict):
                for task in tasks:
                    val = value.get(task)
                    if val is not None:
                        zs_averages[task].append(float(val))
            else:
                zs_averages["all"].append(float(value))

    zs_averages = {
        task: sum(vals) / len(vals) if vals else 0.0
        for task, vals in zs_averages.items()
    }

    grouped[0] = pd.Series(zs_averages)
    grouped = grouped[ks]

    grouped.to_csv(out_path / f"{RESULTS_STRING}_taskwise_grouped_bars.dat", sep="\t")

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.15
    x = range(len(tasks))
    offsets = [-2, -1, 0, 1, 2]
    colors = ['royalblue', 'lightcoral', 'mediumseagreen', 'mediumpurple', 'gold']
    hatches = ['//', 'xx', '', '..', '++']

    for i, (k, label, color, hatch) in enumerate(zip(ks, labels, colors, hatches)):
        values = [grouped.loc[task, k] if task in grouped.index else 0 for task in tasks]
        ax.bar(
            [xi + offsets[i] * bar_width for xi in x],
            values,
            width=bar_width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Micro $F_1$-score")
    ax.set_xlabel("Auxiliary information")
    ax.set_ylim(0, max(grouped.max()) * 1.2)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out_path / f"{RESULTS_STRING}_taskwise_grouped_bars.png")
    print("Saved taskwise grouped bar chart and .dat to", out_path)


def main():
    f1_data = load_aggregated_results()

    rows = []
    for sent, annots in f1_data.items():
        for annotator, ks in annots.items():
            for k, tasks in ks.items():
                for task, score in tasks.items():
                    rows.append({
                        "sentence": sent,
                        "annotator": annotator,
                        "k": int(k.split("=")[1]),
                        "task": task,
                        "f1": float(score)
                    })

    df = pd.DataFrame(rows)

    avg_ak = df.groupby(["annotator", "k"], as_index=False)["f1"].mean()
    baseline_per_annotator = load_avg_baseline_f1_per_annotator(OUT_DIR, RESULTS_STRING)

    for annot, avg_f1 in baseline_per_annotator.items():
        avg_ak = pd.concat([
            avg_ak,
            pd.DataFrame([{
                "annotator": annot,
                "k": 0,
                "f1": avg_f1
            }])
        ], ignore_index=True)

    pivot = avg_ak.pivot(index="annotator", columns="k", values="f1") \
        .reindex(sorted(avg_ak["annotator"].unique()), axis=0) \
        .sort_index(axis=1)

    pivot[[0, 1, 5, 10, 15]].to_csv(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_annotator_and_k.dat", sep="\t")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    pivot[[0, 1, 5, 10, 15]].plot.bar(ax=ax1, rot=0)
    ax1.set_title("Average F1 by Annotator & k (including baseline)")
    ax1.set_xlabel("Annotator")
    ax1.set_ylabel("Avg Micro F1")
    ax1.legend(title="k", bbox_to_anchor=(1, 1))
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_annotator_and_k.png")

    df5 = df[df["k"] == 5]
    avg5 = df5.groupby("annotator")["f1"].mean().sort_index()
    avg5.to_csv(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_annotator_k5.dat", sep="\t")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    avg5.plot.bar(ax=ax2)
    ax2.set_title("Average F1 by Annotator (k=5)")
    ax2.set_xlabel("Annotator")
    ax2.set_ylabel("Avg micro F1")
    ax2.set_ylim(0, 1)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_annotator_k5.png")

    avg_task = df5.groupby("task")["f1"].mean().sort_values(ascending=False)
    avg_task.to_csv(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_task_k5.dat", sep="\t")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    avg_task.plot.bar(ax=ax3)
    ax3.set_title("Average F1 by Task (k=5)")
    ax3.set_xlabel("Task")
    ax3.set_ylabel("Avg micro F1")
    ax3.set_ylim(0, 1)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_task_k5.png")

    df_nz = df[df["k"] != 0]
    avg_at = df_nz.groupby(["annotator", "task"])["f1"].mean().unstack("task")
    avg_at.to_csv(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_annotator_and_task.dat", sep="\t")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    avg_at.plot.bar(ax=ax4)
    baseline = df[df["k"] == 0]["f1"].mean()
    ax4.axhline(baseline, color="gray", linestyle="--", linewidth=1.5)
    ax4.text(-0.4, baseline + 0.005, f"Zero-shot baseline = {baseline:.3f}", color="gray")
    ax4.set_title("Average F1 by Annotator & SEAT Tasks")
    ax4.set_xlabel("Annotator")
    ax4.set_ylabel("Avg micro F1")
    ax4.legend(title="Task", bbox_to_anchor=(1, 1))
    fig4.tight_layout()
    fig4.savefig(OUT_DIR / f"{RESULTS_STRING}_avg_f1_by_annotator_and_task.png")

    plot_taskwise_grouped_bars(
        df,
        OUT_DIR,
        OUT_DIR / f"F1_{RESULTS_STRING}_k0_baseline.json"
    )

    print("Saved all plots and .dat files to:", OUT_DIR)


if __name__ == "__main__":
    main()
