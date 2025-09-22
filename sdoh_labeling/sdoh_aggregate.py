import pandas as pd
from pathlib import Path

# --- Inputs (your paths/columns) ---
agg_path = "/Users/minhle/Documents/HIVE/hospital-comments/sdoh_labeling/majority_vote.csv"
llms4sdoh_path = "/Users/minhle/Documents/HIVE/hospital-comments/sdoh_labeling/llms4sdoh.csv"

id_col = "rec"
phi_col = "phi3_label"
gemma_col = "gemma_label"
llama_col = "llama3_label"
agg_col = "Voted_Result"
llms4sdoh_col = "LLama_SDoH_Labels"

# --- 12-way ontology (target) ---
ONTO12 = [
    "disability-status",
    "employment-status",
    "violence-abuse",
    "substance-use",
    "education",
    "physical-activity",
    "no_sdoh",
    "transportation",
    "living-condition",
    "social-connection",
    "financial-constraint",
    "sexual-activity"
]

# --- Mapping from llms4sdoh atomic labels (21) to the 12-way ontology ---
MAP_21_TO_12 = {
    "PhysSexAbuse": "violence-abuse",
    "AdverseChildhood": "violence-abuse",  # adjust if you prefer not to fold ACEs into violence/abuse
    "Alcohol": "substance-use",
    "Drug": "substance-use",
    "Smoking": "substance-use",
    "EmploymentStatus": "employment-status",
    "EducationLevel": "education",
    "PhysicalActivity": "physical-activity",
    "LivingStatus": "living-condition",
    "EnvironExposure": "living-condition",
    "FinancialIssues": "financial-constraint",
    "Insurance": "financial-constraint",
    "FoodInsecurity": "financial-constraint",
    "MaritalStatus": "social-connection",
    "SocialSupport": "social-connection",
    "Isolation": "social-connection",
    "SexualOrientation": "sexual-activity",
    # Intentionally not mapping: Race, BirthSex, SexualOrientation, LocationBornRaised
    # "nonSDoH" handled specially below
}

def split_labels_21(s):
    if pd.isna(s):
        return set()
    return {t.strip() for t in str(s).split(";") if t.strip()}

def map_21_to_12(labels21):
    mapped = set()
    unmapped = set()
    for lab in labels21:
        if lab == "nonSDoH":
            continue  # handle at end
        target = MAP_21_TO_12.get(lab)
        if target:
            mapped.add(target)
        else:
            unmapped.add(lab)
    if not mapped and ("nonSDoH" in labels21):
        mapped.add("no_sdoh")
    return mapped, unmapped

def reconcile_union(agg_label, llms_set12):
    """Union of agg 1-label and llms multi-label (both in 12-way).
       - If any real labels exist, drop 'no_sdoh'.
       - If none exist and agg is no_sdoh, keep 'no_sdoh'.
    """
    out = set(llms_set12)
    a = (str(agg_label).strip() if pd.notna(agg_label) else "")
    if a:
        out.add(a)
    if "no_sdoh" in out and len(out) > 1:
        out.discard("no_sdoh")
    return out

def pack_labels(s):
    return ";".join(sorted(s)) if s else ""

# --- Load data ---
agg = pd.read_csv(agg_path, low_memory=False)
llms = pd.read_csv(llms4sdoh_path, low_memory=False)

# Keep only needed cols from llms side
llms = llms[[id_col, llms4sdoh_col]].copy().rename(columns={llms4sdoh_col: "llms4sdoh_raw"})

# Inner join on rec; preserves ALL agg columns by default
merged = agg.merge(llms, on=id_col, how="inner")

# Build sets
merged["llms4sdoh_set21"] = merged["llms4sdoh_raw"].apply(split_labels_21)
mapped_unmapped = merged["llms4sdoh_set21"].apply(map_21_to_12)
merged["llms4sdoh_set12"] = mapped_unmapped.apply(lambda t: t[0])
merged["llms4sdoh_unmapped_set"] = mapped_unmapped.apply(lambda t: t[1])

# Reconcile final 12-way multi-label set
merged["final_set12"] = merged.apply(lambda r: reconcile_union(r[agg_col], r["llms4sdoh_set12"]), axis=1)

# Strings for storage
merged["llms4sdoh_mapped_12"] = merged["llms4sdoh_set12"].apply(pack_labels)
merged["llms4sdoh_unmapped"] = merged["llms4sdoh_unmapped_set"].apply(pack_labels)
merged["final_labels_12"] = merged["final_set12"].apply(pack_labels)

# Optional: one-hot indicators for analysis
for cat in ONTO12:
    merged[f"final_{cat}"] = merged["final_set12"].apply(lambda s: int(cat in s))

# Clean up helper set columns (keep readable strings)
merged = merged.drop(columns=["llms4sdoh_set21", "llms4sdoh_set12", "llms4sdoh_unmapped_set", "final_set12"])

# --- Write output ---
out_path = Path(agg_path).with_name("sdoh_final.csv")
merged.to_csv(out_path, index=False)
print(f"Wrote: {out_path} | Rows: {len(merged)}")







