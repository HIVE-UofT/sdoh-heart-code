import pandas as pd
from pathlib import Path

# --- inputs (from your message) ---
sdoh_path = "/Users/minhle/Documents/HIVE/hospital-comments/sdoh_labeling/sdoh_merged_final.csv"
heart_path = "/Users/minhle/Documents/HIVE/hospital-comments/cardiac_labeling/sft_final_heart_labelled_comments.csv"

id_col = "rec"
heart_label = "is_heart_patient"     # column in heart file

# --- helpers ---
def coerce_bool(x):
    """Convert common truthy/falsey values to bool, else return pd.NA."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    # Try numeric
    try:
        return bool(int(float(s)))
    except Exception:
        return pd.NA

# --- load ---
sdoh = pd.read_csv(sdoh_path, low_memory=False)
heart = pd.read_csv(heart_path, low_memory=False)

# normalize ids
sdoh[id_col] = sdoh[id_col].astype(str).str.strip()
heart[id_col] = heart[id_col].astype(str).str.strip()

# coerce heart label to boolean (nullable) and de-duplicate on rec
if heart_label not in heart.columns:
    raise KeyError(f"'{heart_label}' not found in heart file columns: {list(heart.columns)[:20]}...")

heart["_is_heart_bool"] = heart[heart_label].apply(coerce_bool)

# If multiple rows per rec, treat patient as heart patient if ANY row says True
heart_agg = (
    heart.groupby(id_col, as_index=False)["_is_heart_bool"]
         .agg(lambda s: True if (s == True).any() else (False if (s == False).any() else pd.NA))
         .rename(columns={"_is_heart_bool": heart_label})
)

# --- merge: keep ALL SDoH columns + heart label; INNER join for intersection ---
out = sdoh.merge(heart_agg[[id_col, heart_label]], on=id_col, how="inner")

# optional: put the new column near the end (already is) or right after rec:
# cols = list(out.columns)
# cols.insert(cols.index(id_col)+1, cols.pop(cols.index(heart_label)))
# out = out[cols]

# --- save next to the SDoH file ---
out_path = str(Path(sdoh_path).with_name("sdoh_herat_final.csv"))
out.to_csv(out_path, index=False)
print(f"Wrote {len(out):,} rows to {out_path}")
