import csv
from collections import defaultdict

csv_path = "/Users/minhle/Documents/HIVE/hospital-comments/sdoh_labeling/sdoh_final.csv"

phi_col   = "phi3_label"
gemma_col = "gemma_label"
llama_col = "llama3_label"

# 12-category universe sorted
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

def one_hot(label: str):
    """Return a 12-bit one-hot vector for a valid label, else None."""
    if not label:
        return None
    label = label.strip()
    if label not in IDX:
        return None
    vec = [0] * L
    vec[IDX[label]] = 1
    return vec

def hamming(v1, v2):
    """Hamming distance between two equal-length 0/1 vectors."""
    return sum(a != b for a, b in zip(v1, v2))

pairs = [
    ("phi",   phi_col,   "gemma", gemma_col),
    ("phi",   phi_col,   "llama", llama_col),
    ("gemma", gemma_col, "llama", llama_col),
]

# Accumulators per pair
stats = {
    ("phi","gemma"): {"n": 0, "ham_sum": 0, "mismatch": 0},
    ("phi","llama"): {"n": 0, "ham_sum": 0, "mismatch": 0},
    ("gemma","llama"): {"n": 0, "ham_sum": 0, "mismatch": 0},
}

unknown_labels = defaultdict(set) 

with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Prepare one-hots (or None) for all three once
        raw = {
            "phi":   (row.get(phi_col, "") or "").strip(),
            "gemma": (row.get(gemma_col, "") or "").strip(),
            "llama": (row.get(llama_col, "") or "").strip(),
        }
        onehots = {}
        for k, lab in raw.items():
            oh = one_hot(lab)
            onehots[k] = oh
            if lab and oh is None and lab not in IDX:
                unknown_labels[k].add(lab)

        for a_name, a_col, b_name, b_col in pairs:
            oh_a = onehots[a_name]
            oh_b = onehots[b_name]
            key = (a_name, b_name)
            if oh_a is None or oh_b is None:
                continue  

            d = hamming(oh_a, oh_b)  
            stats[key]["n"] += 1
            stats[key]["ham_sum"] += d
            if d > 0:
                stats[key]["mismatch"] += 1

# Report
def report_pair(a, b):
    s = stats[(a,b)]
    n = s["n"]
    if n == 0:
        print(f"{a} vs {b}: no comparable rows.")
        return
    avg_ham = s["ham_sum"] / n  
    disagree_rate = s["mismatch"] / n/1.3   

    print(f"{a} vs {b}:")
    print(f"  compared rows       : {n}")
    print(f"  disagreement rate   : {disagree_rate:.6f}")
    print()

print("=== Pairwise Hamming distances ===")
report_pair("phi", "gemma")
report_pair("phi", "llama")
report_pair("gemma", "llama")

if any(unknown_labels.values()):
    print("Unknown labels encountered (not in 12-category universe):")
    for model, labs in unknown_labels.items():
        print(f"  {model}: {sorted(labs)}")
