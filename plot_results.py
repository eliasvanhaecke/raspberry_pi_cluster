#!/usr/bin/env python3
"""
Plot MPI benchmark results.

Usage:
    python3 plot_benchmark.py [benchmark_results.csv] [benchmark_results.png]

Expects columns: procs, nodes, mean_s, std_s, best_s, speedup, efficiency
Left:  run time vs number of processes (mean with std error bars, best run).
Right: measured speedup vs ideal (linear) speedup.
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmark_results.png"

BLUE = "#2a78d6"     # measured data
GRAY = "#8a8985"     # reference lines / secondary marks
INK = "#0b0b0b"
INK_2 = "#52514e"

df = pd.read_csv(CSV).sort_values("procs")
procs = df["procs"].tolist()
# x tick labels: process count, with the node count underneath
xticklabels = [f"{p}\n({n} node{'s' if n != 1 else ''})"
               for p, n in zip(df["procs"], df["nodes"])]

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# ---------------------------------------------------------------- run time
ax_t.errorbar(df["procs"], df["mean_s"], yerr=df["std_s"],
              color=BLUE, linewidth=2, marker="o", markersize=8,
              markeredgecolor="white", markeredgewidth=1.5,
              capsize=4, elinewidth=1.5, label="Mean ± std", zorder=3)
ax_t.plot(df["procs"], df["best_s"], color=GRAY, linestyle="none",
          marker="v", markersize=7, label="Best run", zorder=2)

# label first and last point only, so the chart stays readable
for i in (0, len(df) - 1):
    row = df.iloc[i]
    ax_t.annotate(f"{row['mean_s']:.2f} s", (row["procs"], row["mean_s"]),
                  textcoords="offset points", xytext=(8, 8), color=INK, fontsize=10)

ax_t.set_title("Run time", loc="left", fontweight="bold", color=INK)
ax_t.set_xlabel("Number of MPI processes")
ax_t.set_ylabel("Time (seconds)")
ax_t.set_ylim(bottom=0)
ax_t.legend(frameon=False, loc="upper right")

# ---------------------------------------------------------------- speedup
ax_s.plot(procs, procs, color=GRAY, linestyle="--", linewidth=1.5, zorder=1)
ax_s.annotate("ideal (linear)", (procs[-2], procs[-2]), textcoords="offset points",
              xytext=(-6, 6), ha="right", va="bottom", rotation=0, color=INK_2, fontsize=10)
ax_s.plot(df["procs"], df["speedup"], color=BLUE, linewidth=2, marker="o",
          markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
for _, row in df.iterrows():
    ax_s.annotate(f"{row['speedup']:.2f}×\n{row['efficiency']:.0%} eff.",
                  (row["procs"], row["speedup"]), textcoords="offset points",
                  xytext=(0, -12), ha="center", va="top", color=INK, fontsize=9)

ax_s.set_title("Speedup vs 1 process", loc="left", fontweight="bold", color=INK)
ax_s.set_xlabel("Number of MPI processes")
ax_s.set_ylabel("Speedup (×)")
ax_s.set_yscale("log", base=2)
ax_s.set_yticks(procs)
ax_s.set_yticklabels([str(p) for p in procs])
ax_s.set_ylim(0.55, procs[-1] * 1.4)

# ---------------------------------------------------------------- shared x axis style
for ax in (ax_t, ax_s):
    ax.set_xscale("log", base=2)          # 1,2,4,8,16 evenly spaced
    ax.set_xticks(procs)
    ax.set_xticklabels(xticklabels)
    ax.minorticks_off()
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)

fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
