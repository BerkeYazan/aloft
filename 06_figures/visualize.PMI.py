# ------------------------- Vis-ΔPMI + Stats.ipynb cell -------------------------
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
# ------------------------- Vis-ΔPMI + Stats.ipynb cell -------------------------
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
import numpy as np

RUN_DIR = Path("data/interim/delta_pmi_full")
CSV     = RUN_DIR / "delta_pmi_scores.csv"

df = pd.read_csv(CSV)

# ---------- Plot ----------
long = df.melt(var_name="TextType", value_name="Delta_PMI").dropna()

plt.figure(figsize=(9, 5))
parts = plt.violinplot(
    [long.loc[long["TextType"] == col, "Delta_PMI"] for col in df.columns],
    showmeans=False, showmedians=False, showextrema=False
)
for pc in parts['bodies']:
    pc.set_alpha(0.4)

plt.boxplot(
    [long.loc[long["TextType"] == col, "Delta_PMI"] for col in df.columns],
    widths=0.12, vert=True, patch_artist=False, showfliers=False
)
plt.xticks(range(1, len(df.columns) + 1), df.columns, rotation=15, ha='right')
plt.ylabel("Δ-PMI (mean per passage)")
plt.title("Lexical-Novelty (Δ-PMI) across text types")
plt.tight_layout()
plt.show()

print("\n=== Δ-PMI descriptive stats ===")
print(long.groupby("TextType")["Delta_PMI"].describe().round(3))

# ---------- Mann-Whitney tests ----------
baseline = "Google Books Length Matched Snippet"
quote_cols = [c for c in df.columns if c != baseline]

def cliffs_delta(x, y):
    """Cliff's Δ: probability that a random x > random y minus the reverse."""
    m, n = len(x), len(y)
    gt = sum((xi > yj) for xi in x for yj in y)
    lt = sum((xi < yj) for xi in x for yj in y)
    return (gt - lt) / (m * n)

print("\n=== Mann-Whitney U (one-sided: quote > snippet) ===")
raw_p = []
for col in quote_cols:
    u, p = mannwhitneyu(df[col].dropna(), df[baseline].dropna(),
                        alternative="greater")
    delta = cliffs_delta(df[col].dropna().values, df[baseline].dropna().values)
    raw_p.append(p)
    print(f"{col:30}  U = {u:>8,.0f}   p = {p:.3e}   CliffΔ = {delta:.3f}")

# ---------- FDR correction ----------
import statsmodels.stats.multitest as multi
rej, p_corr, *_ = multi.multipletests(raw_p, method="fdr_bh")
print("\nBenjamini–Hochberg-corrected p-values:")
for col, pc, r in zip(quote_cols, p_corr, rej):
    mark = "*" if r else ""
    print(f"{col:30}  p_FDR = {pc:.3e} {mark}")
