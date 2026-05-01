import matplotlib.pyplot as plt

COLORS = ['b', 'r', 'g', 'orange', 'purple']
MARKERS = ['o', 's', '^', 'D', 'v']

def plot_accuracies(series, title, save_label):
    """
    Plots one or more Top-r accuracy curves on a single figure.
    :param series: List of (label_str, accuracies_array) tuples.
    :param title: Chart title.
    :param save_label: Output filename (without extension).
    """
    plt.figure(figsize=(10, 6))

    all_accs = []
    max_len = 0
    for i, (label, accuracies) in enumerate(series):
        r_values = range(1, len(accuracies) + 1)
        plt.plot(r_values, accuracies,
                 marker=MARKERS[i % len(MARKERS)],
                 linestyle='-',
                 color=COLORS[i % len(COLORS)],
                 markersize=4,
                 label=label)
        all_accs.extend(accuracies)
        max_len = max(max_len, len(accuracies))

    plt.title(title, fontsize=14)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlim(0, max_len + 1)
    plt.ylim(max(0, min(all_accs) - 2), min(105, max(all_accs) + 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.savefig(f"{save_label}.jpg", dpi=300, bbox_inches='tight')

def plot_roc_b(tprs, fprs, save_label):
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, marker='o', color='b', linestyle='-', markersize=6)
    plt.title('ROC Curve', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{save_label}.jpg', dpi=300, bbox_inches='tight')
