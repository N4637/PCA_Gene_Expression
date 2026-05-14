
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

COLOR_POS  = "#E63946"   
COLOR_NEG  = "#457B9D"   
MARKER_SZ  = 60
ALPHA      = 0.80
FONT_TITLE = 12
FONT_LABEL = 10


print("Loading data ...")

labels = pd.read_csv("data/class.tsv", sep="\t", header=None, names=["label"])
y = labels["label"].values          # shape (105,)  1=ER+, 0=ER-


print("  Parsing columns.tsv.gz ...")
data_lines, header = [], None
with gzip.open("data/columns.tsv.gz", "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
        else:
            parts += [""] * (len(header) - len(parts))
            data_lines.append(parts[:len(header)])

col_map = pd.DataFrame(data_lines, columns=header)
print(f"  col_map : {col_map.shape}  columns: {list(col_map.columns)}")

# 3. LOAD EXPRESSION MATRIX

print("  Loading filtered.tsv.gz ...")
expr = pd.read_csv("data/filtered.tsv.gz", sep="\t", compression="gzip",
                   index_col=None)
expr.columns = expr.columns.str.strip()
print(f"  expr    : {expr.shape[0]} samples x {expr.shape[1]} probes")
print(f"  labels  : {np.sum(y==1)} ER+  |  {np.sum(y==0)} ER-")
print(f"  expr columns sample (first 5): {list(expr.columns[:5])}")


expr_cols_set = set(expr.columns)
best_key_col, best_n = None, 0
for c in col_map.columns:
    n = len(expr_cols_set & set(col_map[c].astype(str)))
    if n > best_n:
        best_n, best_key_col = n, c

print(f"\n  Probe key column : '{best_key_col}'  ({best_n}/{len(col_map)} probes matched)\n")

# Build probe-key -> gene-symbol lookup
id_to_symbol = dict(zip(
    col_map[best_key_col].astype(str),
    col_map["GeneSymbol"].astype(str)
))

# 5. GENE LOOKUP HELPER

symbol_to_probes = {}
for probe_key, symbol in id_to_symbol.items():
    if probe_key in expr_cols_set:
        symbol_to_probes.setdefault(symbol.upper(), []).append(probe_key)

def get_expression(gene_name):
    """Return mean expression across all matched probes for a gene."""
    probes = symbol_to_probes.get(gene_name.upper(), [])
    if not probes:
        nearby = [s for s in symbol_to_probes if gene_name[:3].upper() in s][:10]
        raise KeyError(
            f"Gene '{gene_name}' not found.\n"
            f"  Symbols with similar prefix: {nearby}\n"
            f"  Total symbols in map: {len(symbol_to_probes)}"
        )
    return expr[probes].mean(axis=1).values

xbp1  = get_expression("XBP1")
gata3 = get_expression("GATA3")
print(f"  XBP1  probes: {symbol_to_probes['XBP1']}  range: {xbp1.min():.2f}-{xbp1.max():.2f}")
print(f"  GATA3 probes: {symbol_to_probes['GATA3']}  range: {gata3.min():.2f}-{gata3.max():.2f}")


# 6. PCA on [GATA3, XBP1]

X      = np.column_stack([gata3, xbp1])
X_std  = StandardScaler().fit_transform(X)

pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X_std)
pc1    = X_pca[:, 0]

explained = pca.explained_variance_ratio_
print(f"\n  PCA variance -> PC1: {explained[0]*100:.1f}%   PC2: {explained[1]*100:.1f}%")

loadings = pd.DataFrame(pca.components_.T, index=["GATA3","XBP1"], columns=["PC1","PC2"])
print(f"\n  PC Loadings:\n{loadings.round(4)}\n")


fig = plt.figure(figsize=(13, 5.5))
fig.patch.set_facecolor("#FAFAFA")
gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Figure 1a: XBP1 vs GATA3
for mask, label, color in [
    (y == 1, "ER+", COLOR_POS),
    (y == 0, "ER-", COLOR_NEG),
]:
    ax1.scatter(gata3[mask], xbp1[mask], c=color, s=MARKER_SZ, alpha=ALPHA,
                edgecolors="white", linewidths=0.4, label=label, zorder=3)

ax1.set_xlabel("GATA3 expression", fontsize=FONT_LABEL)
ax1.set_ylabel("XBP1 expression",  fontsize=FONT_LABEL)
ax1.set_title("Figure 1a - XBP1 vs GATA3", fontsize=FONT_TITLE, fontweight="bold")
ax1.legend(fontsize=9, framealpha=0.8, edgecolor="none")
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(True, linestyle="--", alpha=0.35, color="grey")

# Figure 1c: PC1 strip plot
rng = np.random.default_rng(42)
for row_pos, mask, label, color in [
    (1, y == 1, "ER+", COLOR_POS),
    (0, y == 0, "ER-", COLOR_NEG),
]:
    scores  = pc1[mask]
    jitter  = rng.uniform(-0.08, 0.08, size=scores.shape)
    ax2.scatter(scores, row_pos + jitter, c=color, s=MARKER_SZ, alpha=ALPHA,
                edgecolors="white", linewidths=0.4, label=label, zorder=3)

ax2.set_yticks([0, 1])
ax2.set_yticklabels(["ER-", "ER+"], fontsize=FONT_LABEL)
ax2.set_xlabel(f"PC1  ({explained[0]*100:.1f}% variance explained)", fontsize=FONT_LABEL)
ax2.set_title("Figure 1c - Projection onto PC1", fontsize=FONT_TITLE, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.set_ylim(-0.5, 1.5)
ax2.grid(True, axis="x", linestyle="--", alpha=0.35, color="grey")
ax2.legend(fontsize=9, framealpha=0.8, edgecolor="none")

fig.suptitle("GSE5325 - Breast Cancer Gene Expression (Nature Primer, 2008)",
             fontsize=13, fontweight="bold", y=1.02)

plt.savefig("pca_output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved -> pca_output.png")
