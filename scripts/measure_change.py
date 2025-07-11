import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_change_from_baseline(
    baseline_path, comparison_path, ks=None, dimensions=None,
    show_plot=True, output_path="change_from_baseline.png", dat_path="change_from_baseline.dat"):
    if ks is None:
        ks = ['k=1', 'k=5', 'k=10', 'k=15']
    if dimensions is None:
        dimensions = ['all', 'emotion', 'sentiment', 'topic', 'argument']

    with open(baseline_path) as f:
        baseline_data = json.load(f)
    with open(comparison_path) as f:
        full_data = json.load(f)

    changes = defaultdict(lambda: defaultdict(list))
    for entry in baseline_data:
        for annotator in baseline_data[entry]:
            A = set(baseline_data[entry][annotator]['k0_baseline'].split(', '))
            if not A:
                continue
            for k in ks:
                if annotator not in full_data.get(entry, {}):
                    continue
                alt_data = full_data[entry][annotator].get(k, {})
                for dim in dimensions:
                    if dim not in alt_data:
                        continue
                    B = set(alt_data[dim].split(', '))
                    symmetric_diff = A.symmetric_difference(B)
                    change_pct = len(symmetric_diff) / len(A)
                    changes[dim][k].append(change_pct)

    avg_changes = {
        dim: {k: np.mean(changes[dim][k]) for k in ks}
        for dim in dimensions
    }

    with open(dat_path, 'w') as f:
        f.write("x\t1\t5\t10\t15\n")
        for i, dim in enumerate(dimensions):
            values = [avg_changes[dim][f'k={k}'] for k in [1, 5, 10, 15]]
            percent_values = [f"{v:.6f}" for v in values]
            f.write(f"{i}\t" + "\t".join(percent_values) + "\n")

    print(f".dat file saved to: {dat_path}")

    if show_plot:
        bar_width = 0.18
        x = np.arange(len(dimensions))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, k in enumerate(ks):
            values = [avg_changes[dim][k] for dim in dimensions]
            ax.bar(x + i * bar_width, values, width=bar_width, label=k)

        ax.set_xticks(x + bar_width * 1.5)
        ax.set_xticklabels(dimensions, fontsize=12)
        ax.set_ylabel('Change from Baseline (|A Δ B| / |A|)', fontsize=13)
        ax.set_title('Label Change Relative to Baseline (k=0)', fontsize=14)
        ax.legend(title='Model k', fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
        plt.close()

    return avg_changes

def main():
    baseline_path = "../aggregated_results/final_results/final_results_baseline_k0_labels.json"
    comparison_path = "../aggregated_results/final_results/final_results_threshold_3_labels.json"

    ks = ['k=1', 'k=5', 'k=10', 'k=15']
    dimensions = ['all', 'emotion', 'sentiment', 'topic', 'argument']

    avg_changes = compute_change_from_baseline(
        baseline_path=baseline_path,
        comparison_path=comparison_path,
        ks=ks,
        dimensions=dimensions,
        show_plot=True,
        output_path="../aggregated_results/final_results/change/change_from_baseline.png",
        dat_path="../aggregated_results/final_results/change/change_from_baseline.dat"
    )

if __name__ == "__main__":
    main()
