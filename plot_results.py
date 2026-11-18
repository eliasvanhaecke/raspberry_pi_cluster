import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")
df.plot(x="procs", y="mean_s")
plt.savefig('benchmark_results.png')