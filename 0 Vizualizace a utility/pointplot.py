import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12

root = Path("./2025-08-18_test/")
methods = [
    "rnd_0", "sp_ag", "oi-3b", "oi-3b_sp", "lai-3b", "pv-3b", "pv-5b",
]
# methods = [
#     "rnd_0", "rnd_1", "rnd_2", "rnd_3",
#     "sp_ag", "sp_ag_num-c", "sp_ag_num-w",
#     "oi-3b", "oi-3b-qrt", "oi-5b",
#     "oi-3b_sp", "oi-3b-qrt_sp", "oi-5b_sp",
#     "lai-3b", "lai-5b",
#     "lai-3b_sp", "lai-5b_sp",
#     "pv-3b", "pv-5b",
#     "pv-3b_sp", "pv-5b_sp",
# ]
# metrics = ["loss", "mae", "rmse", "rsq", "fscore_sum_0.3", "precis_sum_0.3", "recall_sum_0.3"]
# metrics = ["ap_sum_0.3", "ap_sum_0.5", "loss", "rmse", "rsq"]
metrics = ["ap_sum_0.3"]
yranges = [(0, 1)]
# metrics = ["ap_sum_0.3", "ap_avg_0.3"]
# yranges = [(0, 1), (0, 1)]
# yranges = [(0, 1), (0, 1), (2, 5), (0, 5), (0, 1)]
# metrics = ["amae", "arsq"]
# yranges = [(1, 6), (0, 1)]
# metrics = ["mae", "rmse", "rsq"]
# metrics = ["fscore_sum_0.3", "precis_sum_0.3", "recall_sum_0.3"]

SPACE_IN = 0.05
SPACE_OUT = 0.2
BOX_WIDTH = 0.1
PLOT_X_MULT = 2

def get_csv_metric(dataset, metric):
    csv_path = root / dataset / "log_test.csv"
    csv = pd.read_csv(csv_path, sep=";")
    return csv[metric][0]

def get_pkl_metric(dataset, metric):
    with open(root / dataset / "metrics.pkl", "rb") as fp:
        pick = pickle.load(fp)
    # return pick[metric]
    # return pick['count_per_cf']["0.45"][metric]
    return pick['detection_per_cf'][0.45][metric]

def plot_metric(metric, ylim):
    data = {}
    positions = []
    method_name_positions = []
    
    current_position = 0
    for m in methods:
        method_data = {}
        for t in range(5):
            outer_cv_data = []
            for v in range(5):
                if t == v:
                    continue
                outer_cv_data.append(get_csv_metric(f"{m}-t{t}-v{v}", metric))
                if t == 2 and v == 0:
                    # ale fuj
                    method_name_positions.append(current_position)
            method_data[t+1] = outer_cv_data
            positions.append(current_position)
            current_position += SPACE_IN
        data[m] = method_data
        current_position += SPACE_OUT
    
    
    labels = [x for xs in data.values() for x in xs]
    values = [x for xs in data.values() for x in xs.values()]
    fig, ax = plt.subplots(figsize=[len(methods)*PLOT_X_MULT, 5], dpi=300)
    bp = ax.boxplot(values, positions=positions, widths=BOX_WIDTH, tick_labels=labels)
    for comp in bp.values():
        for line in comp:
            line.remove()
            
    for p, v in zip(positions, values):
        ax.scatter([p] * len(v), v, s=8, c="tab:red", marker="x", alpha=0.7)
    fig.suptitle(f"Hodnoty metriky v rámci vnořených CV pro různé metody rozdělení\nMetrika: {metric}", weight='bold')
    ax.set_ylabel("Hodnota metriky", style='italic')
    ax.set_xlabel("Fold, použitý jako testovací\nMetoda rozložení", fontweight="bold", style='italic', labelpad=15.0)
    ax.set_ylim(ylim)
    ax.grid(which="both", axis="y")
    method_y = ax.get_ylim()[0]
    for m, p in zip(methods, method_name_positions):
        pos = ax.transLimits.transform([p, method_y])
        ax.text(x=pos[0], y=pos[1]-0.1, s=m, horizontalalignment='center', transform=ax.transAxes, fontweight="bold")
    return bp

for m, y in zip(metrics, yranges):
    bp = plot_metric(m, y)