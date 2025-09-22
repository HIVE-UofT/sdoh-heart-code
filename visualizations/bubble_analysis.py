import csv
from collections import defaultdict

csv_path = "./sdoh_heart_final.csv"

sdoh_col = "sdoh_label"

all_labels = defaultdict(int)

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels = row[sdoh_col].split(";") if row[sdoh_col] else []
        for label in labels:
            all_labels[label] += 1

# remove key no_sdoh
if "no_sdoh" in all_labels:
    del all_labels["no_sdoh"]

formatted_labels = {label.replace("-", " ").title(): count for label, count in all_labels.items()}

print(formatted_labels)

# print json like lines
for label, count in formatted_labels.items():
    print(f"{{category:'{label}', value:{count} }},")


heart_sdoh_labels = defaultdict(int)
heart_col = "heart_label"

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row[heart_col] == "TRUE":
            labels = row[sdoh_col].split(";") if row[sdoh_col] else []
            for label in labels:
                heart_sdoh_labels[label] += 1

if "no_sdoh" in heart_sdoh_labels:
    del heart_sdoh_labels["no_sdoh"]

formatted_heart_labels = {label.replace("-", " ").title(): count for label, count in heart_sdoh_labels.items()}

print(formatted_heart_labels)

# print json like lines
for label, count in formatted_heart_labels.items():
    print(f"{{category:'{label}', value:{count} }},")
