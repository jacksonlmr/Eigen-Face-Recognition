import numpy as np
from plot import plot_accuracies

series = [
    ("t=0.80", np.load("results_80/accuracies_80.npy")),
    ("t=0.90", np.load("results_90/accuracies_90.npy")),
    ("t=0.95", np.load("results_95/accuracies_95.npy")),
]

plot_accuracies(series, "CMC Curves", "Experiment_II")
