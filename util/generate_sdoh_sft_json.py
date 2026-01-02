"""
input file: merged_sdoh_labeled.csv

each line in the input file has the following columns:
rec,Comment,no_sdoh,financial-constraint,living-condition,transportation,education,employment-status,social-connection,violence-abuse,disability-status,physical-activity,substance-use,discrimination

example:
41157,Very attentive nursing staff made our stay great.,True,False,False,False,False,False,False,False,False,False,False,False

rec: unique record identifier
Comment: patient comment text
other columns: True if the label applies, False if not

If the comment has no sdoh, then the no_sdoh column is True and all other label columns are False.
If all label columns are False, then the comment has not been labeled and should not be included in the output file.
In other words, only include comments that have at least label being True (including no_sdoh).



output file: sdoh_sft.jsonl

each line in the output file should be a json object with the following structure:

{
    "instruction": 
    "<INSTRUCTION>",
    "input": "<comment>",
    "output": "<labels comma separated>"

}

"""

import pandas as pd
import json
from textwrap import dedent

input_file = "sdoh_human_labels.csv"
output_file = "sdoh_human_sft.jsonl"

LABEL_COLUMNS = [
    "financial-constraint",
    "living-condition",
    "transportation",
    "education",
    "employment-status",
    "social-connection",
    "violence-abuse",
    "disability-status",
    "physical-activity",
    "substance-use",
    "sexual-activity",
    "discrimination",
    "no_sdoh",
]

INSTRUCTION = dedent("""You are a strict multi-label classifier for Social Determinants of Health (SDOH).

Goal:
- Identify labels only if the patient comment reveals underlying life circumstances or social factors
- These factors do not have to have direct impacts on health or current medical care, but should reflect ongoing social conditions affecting the patient's life
- DO NOT label issues that arise solely from current medical care, treatment, discharge instructions, surgery effects, temporary recovery, hospital processes, or wait times.

Allowed labels:
financial-constraint, living-condition, transportation, education, employment-status,
social-connection, violence-abuse, disability-status, physical-activity, substance-use,
sexual-activity, discrimination, no_sdoh

These are the definition for each label:

- financial-constraint: A state of limited, uncertain, or unstable access to sufficient financial resources to meet current and ongoing needs such as housing, food, utilities, and medical care.
- living-condition: Circumstances related to the safety, stability, adequacy, and affordability of one's housing and surrounding environment, including experiences of housing instability, homelessness, or inadequate housing.
- transportation: Uncertain, limited, or no access to safe, reliable, accessible, affordable, and socially acceptable means of transportation necessary to maintain health, well-being, or livelihood.
- education: The attainment of formal education and literacy skills necessary for employment, health decision-making, and daily functioning, including having less than a high school education or limited health literacy.
- employment-status: Access to stable, safe, and adequate employment that provides sufficient income and benefits; retired status is included only when it affects financial stability.
- social-connection: The structural, functional, and quality aspects of an individual's social interactions—including social networks, support, isolation, perceived loneliness, and barriers such as language limitations.
- violence-abuse: Exposure to physical, sexual, psychological, or financial harm, coercion, or neglect within relationships of trust or intimacy, including intimate partner violence or elder abuse.
- disability-status: A physical, sensory, cognitive, or mental impairment—including conditions such as significant pain, allergies, morbid obesity, or temporary wheelchair use—that limits daily functioning or participation in life activities.
- physical-activity: Engagement in bodily movement that contributes to maintaining or improving health, physical fitness, and overall well-being; insufficient activity may increase risk for chronic disease and reduced quality of life. 
- substance-use: The use of alcohol, tobacco, prescription medications, or illicit substances in a manner that may negatively impact health, functioning, or social well-being.
- discrimination: Differences in access to resources, opportunities, or care based on personal characteristics such as age, sex, gender identity, sexual orientation, race, ethnicity, or other identity factors.

Instructions:
- Output one or more labels from the allowed list, separated by commas with no extra words.
- If none apply, output: no_sdoh
- Do not invent labels. Do not add explanations. Only return labels.


Examples (format only; content varies):
- "I can't afford my medication." -> financial-constraint
- "It's a very long 2-hour bus ride to the hospital." -> transportation
- "After surgery I couldn't drive for 3 days." -> no_sdoh
""").strip()


def is_true(value):
    """Robust check for True values (handles bools, 'True'/'False', 1/0)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def main():
    df = pd.read_csv(input_file)

    # Make sure required columns exist
    missing = [c for c in ["Comment"] + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            comment = str(row["Comment"]).strip()

            # Skip empty comments
            if not comment:
                continue

            # Collect labels marked True
            true_labels = [col for col in LABEL_COLUMNS if is_true(row[col])]

            # Skip rows with no labels at all (unlabeled)
            if not true_labels:
                continue

            # Enforce consistency:
            # If any *specific* SDOH label is present, drop no_sdoh
            specific_labels = [lbl for lbl in true_labels if lbl != "no_sdoh"]
            if specific_labels:
                labels_final = specific_labels
            else:
                # Only no_sdoh is true
                labels_final = ["no_sdoh"]

            # Format output: comma-separated, no spaces
            output_labels = ",".join(labels_final)

            record = {
                "instruction": INSTRUCTION,
                "input": comment,
                "output": output_labels,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
