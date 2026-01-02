






# convert excel files to csv

# excel files: 10_zahra.xlsx   1_hemn.xlsx     2_omid.xlsx     3_sepehr.xlsx   4_chris.xlsx    5_amir.xlsx     6_zhenxiao.xlsx 7_fang.xlsx     8_javier.xlsx   9_minh.xlsx




# import pandas as pd

# excel_files = [
#     "1_hemn.xlsx", "2_omid.xlsx", "3_sepehr.xlsx", "4_chris.xlsx", "5_amir.xlsx",
#     "6_zhenxiao.xlsx", "7_fang.xlsx", "8_javier.xlsx", "9_minh.xlsx", "10_zahra.xlsx"
# ]

# for excel_file in excel_files:
#     df = pd.read_excel(excel_file)
    
#     csv_file = excel_file.replace('.xlsx', '.csv')
#     df.to_csv(csv_file, index=False)
    

# # Merging all CSV files into one
# merged_csv_file = "sdoh_human_labels.csv"
# with open(merged_csv_file, 'w') as outfile:
#     for i, excel_file in enumerate(excel_files):
#         csv_file = excel_file.replace('.xlsx', '.csv')
#         with open(csv_file, 'r') as infile:
#             if i != 0:
#                 next(infile)  # Skip header row for all but the first file
#             outfile.write(infile.read())
# print(f"Merged all CSV files into {merged_csv_file}")







# # read csv file
# CSV_INPUT_1 = "/Users/minhle/Documents/HIVE/sdoh-heart-code/sdoh_human_labels/4_chris.csv"
# CSV_INPUT_2 = "/Users/minhle/Documents/HIVE/sdoh-heart-code/chris_gpt_sdoh_labels.csv"
# CSV_INPUT_3 = "/Users/minhle/Documents/HIVE/sdoh-heart-code/chris_gpt_sdoh_labels_IMPROVED.csv"

# CSV_OUTPUT = "chris_gpt_compare.csv"

# # Join Chris's labels and GPT file into comparison file
# # Current Chris's file (CSV_INPUT_1) contains columns: rec, Comment, no_sdoh, financial-constraint, living-condition,
# # transportation, education, employment-status, social-connection, violence-abuse, 
# # disability-status, physical-activity, substance-use, discrimination
# # each sdoh column has either "True" or "False"

# # CSV_INPUT_2 has 3 columns: rec, Comment, gpt_label
# # chris_label should be a comma-separated list of True labels from Chris's file

# # CSV_INPUT_3 has 3 columns: rec, Comment, gpt_label_IMPROVED
# # output file should have rec, Comment, chris_label, gpt_label, gpt_label_IMPROVED

# # The three files will have identical order of 'rec' values, so we can join on that directly
# import csv


# # Columns in Chris's file that represent SDOH labels
# CHRIS_LABEL_COLUMNS = [
#     "no_sdoh",
#     "financial-constraint",
#     "living-condition",
#     "transportation",
#     "education",
#     "employment-status",
#     "social-connection",
#     "violence-abuse",
#     "disability-status",
#     "physical-activity",
#     "substance-use",
#     "discrimination",
# ]


# def is_true(value) -> bool:
#     """
#     Return True if the value looks like a True boolean from CSV.
#     Handles actual booleans and common string representations.
#     """
#     if isinstance(value, bool):
#         return value
#     if value is None:
#         return False
#     s = str(value).strip().lower()
#     return s in {"true", "t", "1", "yes", "y"}


# def load_gpt_labels(path, label_column_name, label_key_name):
#     gpt_by_rec = {}
#     with open(path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             rec = row.get("rec")
#             if rec is None:
#                 continue
#             gpt_by_rec[rec] = {
#                 "Comment": row.get("Comment", ""),
#                 label_key_name: row.get(label_column_name, ""),
#             }
#     return gpt_by_rec

# def main():
#     gpt_data = load_gpt_labels(CSV_INPUT_2, "gpt_label", "gpt_label")
#     gpt_data_improved = load_gpt_labels(CSV_INPUT_3, "gpt_label_IMPROVED", "gpt_label_IMPROVED")

#     with open(CSV_INPUT_1, newline="", encoding="utf-8") as f_in, \
#          open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f_out:

#         reader = csv.DictReader(f_in)

#         fieldnames = ["rec", "Comment", "chris_label", "gpt_label", "gpt_label_IMPROVED"]
#         writer = csv.DictWriter(f_out, fieldnames=fieldnames)
#         writer.writeheader()

#         for row in reader:
#             rec = row.get("rec")
#             comment = row.get("Comment", "")

#             true_labels = [
#                 col for col in CHRIS_LABEL_COLUMNS
#                 if is_true(row.get(col, False))
#             ]
#             chris_label_str = ",".join(true_labels)

#             gpt_entry = gpt_data.get(rec, {})
#             gpt_label = gpt_entry.get("gpt_label", "")

#             gpt_entry_improved = gpt_data_improved.get(rec, {})
#             gpt_label_improved = gpt_entry_improved.get("gpt_label_IMPROVED", "")

#             if not comment:
#                 comment = gpt_entry.get("Comment", "")

#             writer.writerow({
#                 "rec": rec,
#                 "Comment": comment,
#                 "chris_label": chris_label_str,
#                 "gpt_label": gpt_label,
#                 "gpt_label_IMPROVED": gpt_label_improved,
#             })

# if __name__ == "__main__":
#     main()
