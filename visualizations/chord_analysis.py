import csv
import itertools
from collections import Counter
from pathlib import Path

MIN_COUNT = 1500
# === Config ===
csv_path = "./sdoh_heart_final.csv"
sdoh_col = "sdoh_label"



# Canonical SDOH categories (exclude 'no_sdoh')
CATS = [
    "social-connection",
    "employment-status",
    "transportation",
    "violence-abuse",
    "financial-constraint",
    "physical-activity",
    "sexual-activity",
    "education",
    "living-condition",
    "substance-use",
    "disability-status",
]
IDX = {c: i for i, c in enumerate(CATS)}
N = len(CATS)
category_counts = {c: 0 for c in CATS}
# === Build co-occurrence matrix ===
# Symmetric matrix; diagonal counts single-label rows
co_mat = [[0 for _ in range(N)] for _ in range(N)]

total_rows = 0
used_rows = 0
skipped_no_sdoh = 0
skipped_empty = 0
warn_unknown = Counter()

with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['heart_label'] == "TRUE":
            total_rows += 1
            raw = (row.get(sdoh_col, "") or "").strip()
            if not raw:
                skipped_empty += 1
                continue

            labels = [s.strip() for s in raw.split(";") if s.strip()]
            # Exclude any row that contains 'no_sdoh'
            if any(l == "no_sdoh" for l in labels):
                skipped_no_sdoh += 1
                continue

            # Keep only canonical labels; track unknowns for visibility
            valid = [l for l in labels if l in IDX]
            for l in labels:
                if l not in IDX and l != "no_sdoh":
                    warn_unknown[l] += 1

            if not valid:
                # nothing to count
                continue
            for l in set(valid):  # set() so a label in a row is counted once
                category_counts[l] += 1

            used_rows += 1

            # Single-label row -> add to diagonal
            if len(valid) == 1:
                i = IDX[valid[0]]
                co_mat[i][i] += 1
            else:
                # Multi-label row -> add 1 to each unordered pair
                # (treat as undirected co-occurrence)
                seen = sorted(set(valid))  # de-dup within a row
                for a, b in itertools.combinations(seen, 2):
                    ia, ib = IDX[a], IDX[b]
                    co_mat[ia][ib] += 1
                    co_mat[ib][ia] += 1

# === Report quick stats ===
print(f"Total rows: {total_rows}")
print(f"Used (excluding no_sdoh/empty): {used_rows}")
print(f"Skipped empty: {skipped_empty}")
print(f"Skipped with 'no_sdoh': {skipped_no_sdoh}")
if warn_unknown:
    print("Unknown labels encountered (ignored):")
    for k, v in warn_unknown.most_common():
        print(f"  {k}: {v}")

# === Try to draw a chord diagram ===
# If the library isn't present, we save matrix & labels for external tools.
def save_matrix_files():
    import csv as _csv
    Path("out").mkdir(exist_ok=True)
    with open("out/heart_sdoh_chord_matrix.csv", "w", newline="", encoding="utf-8") as out_f:
        w = _csv.writer(out_f)
        w.writerow([""] + CATS)
        for i, row in enumerate(co_mat):
            w.writerow([CATS[i]] + row)
    with open("out/heart_sdoh_labels.txt", "w", encoding="utf-8") as lf:
        lf.write("\n".join(CATS))
    print("Saved matrix to out/heart_sdoh_chord_matrix.csv and labels to out/heart_sdoh_labels.txt")

# stylized masked names: financial-constraint -> Financial Constraint
names_masked = [c.replace("-", " ").title() if category_counts[c] >= MIN_COUNT else "" for c in CATS]   

try:
    from mpl_chord_diagram import chord_diagram
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10), dpi=160)
    ax = plt.gca()

    # diag=True shows the diagonal (self) ribbons for single-label rows.
    # sort='size' groups larger connections; gap controls spacing between sectors.
    chord_diagram(
        co_mat,
        names=names_masked,
        ax=ax,
        sort="size",
        gap=0.03,
        diag=True
    )

    plt.title("SDOH Co-occurrence Chord Diagram (excluding no_sdoh)")
    plt.tight_layout()
    Path("out").mkdir(exist_ok=True)
    outfile = "out/heart_sdoh_chord.png"
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved chord diagram to {outfile}")
except ImportError:
    print("mpl-chord-diagram not installed; skipping plot.")
    print("Install with: pip install mpl-chord-diagram matplotlib")
    save_matrix_files()
