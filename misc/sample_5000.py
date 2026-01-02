# sample_5000_comments.py
import csv
import random

# Path to full dataset
CSV_INPUT = "/Users/minhle/Documents/HIVE/sdoh-heart-code/cardiac_labeling/heart_final.csv"
CSV_AVOID_1 = "/Users/minhle/Documents/HIVE/sdoh-heart-code/sdoh_human_labels/sdoh_human_labels.csv"
CSV_AVOID_2 = "/Users/minhle/Documents/HIVE/sdoh-heart-code/gpt_sdoh_label_4930.csv"
# Output file containing only a 5000-row sample with rec + Comment
CSV_SAMPLE_OUTPUT = "sample_70.csv"

REC_COL = "rec"
COMMENT_COL = "Comment"


def main():
    # 1) Load rec IDs to avoid
    avoid_recs = set()
    with open(CSV_AVOID_1, newline="", encoding="utf-8") as f_avoid:
        avoid_reader = csv.DictReader(f_avoid)
        if REC_COL not in (avoid_reader.fieldnames or []):
            raise ValueError(f"Avoid CSV must contain a '{REC_COL}' column.")
        for row in avoid_reader:
            rec_id = row.get(REC_COL)
            if rec_id is not None:
                avoid_recs.add(str(rec_id).strip())

    with open(CSV_AVOID_2, newline="", encoding="utf-8") as f_avoid2:
        avoid_reader2 = csv.DictReader(f_avoid2)
        if REC_COL not in (avoid_reader2.fieldnames or []):
            raise ValueError(f"Avoid CSV must contain a '{REC_COL}' column.")
        for row in avoid_reader2:
            rec_id = row.get(REC_COL)
            if rec_id is not None:
                avoid_recs.add(str(rec_id).strip())

    print(f"[INFO] Loaded {len(avoid_recs)} rec IDs to avoid.")

    # 2) Load full dataset
    print(f"[INFO] Reading input from {CSV_INPUT!r}...")
    with open(CSV_INPUT, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if COMMENT_COL not in (reader.fieldnames or []):
            raise ValueError(f"Input CSV must contain a '{COMMENT_COL}' column.")
        if REC_COL not in (reader.fieldnames or []):
            raise ValueError(f"Input CSV must contain a '{REC_COL}' column.")

        rows = list(reader)

    total_rows = len(rows)
    print(f"[INFO] Loaded {total_rows} rows from {CSV_INPUT}.")

    # 3) Filter out rows whose rec is in avoid_recs
    eligible_rows = []
    for idx, row in enumerate(rows, start=1):
        rec_id = row.get(REC_COL)
        if rec_id is None or str(rec_id).strip() == "":
            # fallback to local index if rec is missing
            rec_id_str = str(idx)
        else:
            rec_id_str = str(rec_id).strip()

        if rec_id_str in avoid_recs:
            continue  # skip this row

        # store normalized rec in the row so we don't recompute later
        row["_normalized_rec"] = rec_id_str
        eligible_rows.append(row)

    print(f"[INFO] {len(eligible_rows)} rows remain after excluding avoid list.")

    if not eligible_rows:
        raise ValueError("No eligible rows left after filtering rec IDs to avoid.")

    # 4) Sample up to 70 rows from eligible_rows
    sample_size = min(70, len(eligible_rows))
    sample_rows = random.sample(eligible_rows, sample_size)
    print(f"[INFO] Sampling {sample_size} rows for separate labeling.")

    # 5) Write output CSV with rec + Comment
    with open(CSV_SAMPLE_OUTPUT, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[REC_COL, COMMENT_COL])
        writer.writeheader()

        for row in sample_rows:
            rec_id = row.get("_normalized_rec")
            if rec_id is None or str(rec_id).strip() == "":
                rec_id = ""  # should not happen, but just in case

            comment = (row.get(COMMENT_COL, "") or "").replace("\n", " ").replace("\r", " ").strip()

            writer.writerow({
                REC_COL: rec_id,
                COMMENT_COL: comment,
            })

    print(f"[DONE] Wrote {sample_size} sampled comments (excluding avoid recs) to {CSV_SAMPLE_OUTPUT!r}")


if __name__ == "__main__":
    main()
