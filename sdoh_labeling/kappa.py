import csv
from collections import Counter
from math import isnan

csv_path = "/Users/minhle/Documents/HIVE/hospital-comments/sdoh_labeling/sdoh_final.csv"

phi_col   = "phi3_label"
gemma_col = "gemma_label"
llama_col = "llama3_label"

# 12-class universe (keep consistent across scripts)
LABELS = [
    "disability-status",
    "education",
    "employment-status",
    "financial-constraint",
    "living-condition",
    "no_sdoh",
    "physical-activity",
    "sexual-activity",
    "social-connection",
    "substance-use",
    "transportation",
    "violence-abuse",
]
IDX = {lab: i for i, lab in enumerate(LABELS)}
L = len(LABELS)

def init_conf():
    return [[0 for _ in range(L)] for __ in range(L)]

def update_pair(conf, marg_a, marg_b, a_lab, b_lab):
    """Update confusion and marginals if both labels are valid."""
    if a_lab in IDX and b_lab in IDX:
        ia, ib = IDX[a_lab], IDX[b_lab]
        conf[ia][ib] += 1
        marg_a[a_lab] += 1
        marg_b[b_lab] += 1
        return True
    return False

def kappa_from_conf(conf, marg_a, marg_b):
    n = sum(sum(r) for r in conf)
    if n == 0:
        return dict(n=0, Po=float('nan'), Pe=float('nan'), kappa=float('nan'),
                    exp_hamming_by_chance=float('nan'))
    agree = sum(conf[i][i] for i in range(L))
    Po = agree / n
    pa = {c: marg_a.get(c, 0) / n for c in LABELS}
    pb = {c: marg_b.get(c, 0) / n for c in LABELS}
    Pe = sum(pa[c] * pb[c] for c in LABELS)
    kappa = (Po - Pe) / (1 - Pe) if (1 - Pe) > 0 else float('nan')
    exp_ham = 2 * (1 - Pe)  # expected Hamming in your 0/2-per-row setup
    return dict(n=n, Po=Po, Pe=Pe, kappa=kappa, exp_hamming_by_chance=exp_ham)

def print_confusion(conf, normalize=None, title="", max_col_width=9):
    """
    normalize: None | 'row'
      None -> raw counts
      'row' -> each row sums to 100 (%)
    """
    print(title)
    # Header
    col_hdrs = [lab[:max_col_width] for lab in LABELS]
    print(" " * (max_col_width+2) + " | " + " ".join(f"{h:>{max_col_width}}" for h in col_hdrs))
    print("-" * ((max_col_width+2) + 3 + (max_col_width+1)*L))
    # Rows
    for i, row in enumerate(conf):
        if normalize == 'row':
            s = sum(row)
            vals = [(100.0 * v / s) if s else 0.0 for v in row]
            row_str = " ".join(f"{v:>{max_col_width}.1f}" for v in vals)
        else:
            row_str = " ".join(f"{v:>{max_col_width}d}" for v in row)
        print(f"{LABELS[i][:max_col_width]:<{max_col_width}} -> | {row_str}")
    print()

# Prepare structures for three pairs
pairs = [
    ("phi",   phi_col,   "gemma", gemma_col),
    ("phi",   phi_col,   "llama", llama_col),
    ("gemma", gemma_col, "llama", llama_col),
]
confs   = {("phi","gemma"): init_conf(), ("phi","llama"): init_conf(), ("gemma","llama"): init_conf()}
marg_as = {("phi","gemma"): Counter(),    ("phi","llama"): Counter(),    ("gemma","llama"): Counter()}
marg_bs = {("phi","gemma"): Counter(),    ("phi","llama"): Counter(),    ("gemma","llama"): Counter()}

# Build confusion matrices
with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labs = {
            "phi":   (row.get(phi_col, "") or "").strip(),
            "gemma": (row.get(gemma_col, "") or "").strip(),
            "llama": (row.get(llama_col, "") or "").strip(),
        }
        # update for each pair if both valid
        for a_name, a_col, b_name, b_col in pairs:
            conf = confs[(a_name, b_name)]
            ma   = marg_as[(a_name, b_name)]
            mb   = marg_bs[(a_name, b_name)]
            update_pair(conf, ma, mb, labs[a_name], labs[b_name])

# Report
for key in [("phi","gemma"), ("phi","llama"), ("gemma","llama")]:
    a, b = key
    conf = confs[key]
    metrics = kappa_from_conf(conf, marg_as[key], marg_bs[key])
    n = metrics["n"]
    print(f"=== {a} vs {b} ===")
    if n == 0:
        print("No comparable rows.\n")
        continue
    print(f"Compared rows           : {n}")
    print(f"Observed agreement (Po) : {metrics['Po']:.4f}")
    print(f"Chance agreement  (Pe)  : {metrics['Pe']:.4f}")
    print(f"Cohen's kappa           : {metrics['kappa']:.4f}")
    print(f"Exp. Hamming by chance  : {metrics['exp_hamming_by_chance']:.4f}")
    # Confusion tables
    # print_confusion(conf, normalize=None,  title="Confusion (raw counts)")
    print_confusion(conf, normalize='row', title="Confusion (row-normalized %)")
