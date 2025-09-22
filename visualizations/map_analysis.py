import csv
from collections import defaultdict, Counter
import pprint

csv_path = "./sdoh_heart_final.csv"
hospital_col = "Hospital"

# Counts
total_comments = Counter()
total_comments_with_sdoh = Counter()
sdoh_counts = defaultdict(Counter)

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        hospital = (row.get(hospital_col) or "").strip()
        if not hospital:
            continue

        total_comments[hospital] += 1  # Count every comment

        labels = (row.get("sdoh_label") or "").strip()
        if not labels:
            continue

        sdoh_list = [s.strip() for s in labels.split(";") if s.strip()]
        if any(s != "no_sdoh" for s in sdoh_list):
            total_comments_with_sdoh[hospital] += 1  # Only count if at least 1 valid SDOH

        for sdoh in sdoh_list:
            sdoh_counts[hospital][sdoh] += 1

name_map = {
    "Almonte General Hospital": "Almonte General Hospital",
    "Bluewater": "Bluewater Health",
    "Brant Community Healthcare System": "Brant Community Healthcare System",
    "Campbellford": "Campbellford Memorial Hospital",
    "Children's Hospital of Eastern Ontario": "Children's Hospital of Eastern Ontario",
    "Dryden Regional Health Centre Comments": "Dryden Regional Health Centre",
    "Erie Shores": "Erie Shores Healthcare",
    "Georgian Bay": "Georgian Bay General Hospital",
    "Grand River Hospital": "Grand River Hospital",
    "Guelph General Hospital": "Guelph General Hospital",
    "Hamilton": "Hamilton Health Sciences",
    "Health Sciences North": "Health Sciences North",
    "Holland Bloorview Kids Rehabilitation Centre": "Holland Bloorview Kids Rehabilitation Hospital",
    "Joseph Brant Hospital": "Joseph Brant Hospital",
    "Kingston": "Kingston Health Sciences Centre",
    "London": "London Health Sciences Centre",
    "Mackenzie Health": "Mackenzie Health",
    "Markham Stouffville": "Markham Stouffville Hospital",
    "Michael Garron": "Michael Garron Hospital",
    "Mount Sinai": "Mount Sinai Hospital",
    "Muskoka Algonquin Healthcare": "Muskoka Algonquin Healthcare",
    "Niagara": "Niagara Health",
    "Norfolk General Hospital": "Norfolk General Hospital",
    "North York": "North York General Hospital",
    "Orillia": "Orillia Soldiers' Memorial Hospital",
    "Pembroke": "Pembroke Regional Hospital",
    "Peterborough Regional Health Centre": "Peterborough Regional Health Centre",
    "Quinte Health Care Corporation": "Quinte Health Care",
    "Riverside Healthcare": "Riverside Health Care",
    "Royal Victoria": "Royal Victoria Regional Health Centre",
    "Sault Area": "Sault Area Hospital",
    "Scarborough": "Scarborough Health Network",
    "SickKids Hospital": "SickKids Hospital",
    "Southlake": "Southlake Regional Health Centre",
    "St. Michaels": "St Michael's Hospital",
    "St. Josephs": "St. Joseph's Health Centre",
    "St. Joseph's Healthcare Hamilton": "St. Joseph's Healthcare Hamilton",
    "Sunnybrook": "Sunnybrook Health Sciences Centre",
    "Ottawa Hospital": "The Ottawa Hospital",
    "Thunder Bay Regional": "Thunder Bay Regional Health Sciences Centre",
    "Timmins": "Timmins and District Hospital",
    "Trillium": "Trillium Health Partners",
    "University Health Network": "University Health Network",
    "Winchester": "Winchester District Memorial Hospital",
    "Windsor Regional": "Windsor Regional Hospital",
    "Woodstock General Hospital": "Woodstock Hospital"
}

def pretty_label(label: str) -> str:
    return label.replace("-", " ").title()
# Build JSON-like output
final_output = []
for hospital, counts in sdoh_counts.items():
    mapped_name = name_map.get(hospital, hospital)

    counts.pop("no_sdoh", None)  # Remove "no_sdoh"
    top5_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top5 = {pretty_label(k): v for k, v in top5_items}

    final_output.append({
        "hospital": mapped_name,
        "total_comments": total_comments[hospital],
        "total_comments_with_sdoh": total_comments_with_sdoh[hospital],
        "top_5_sdoh": top5
    })

# pprint.pprint(final_output)
print(len(final_output), "hospitals with SDOH data")