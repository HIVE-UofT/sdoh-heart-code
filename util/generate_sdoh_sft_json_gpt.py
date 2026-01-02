"""
input file: gpt_sdoh_label_5000.csv

the input file has the following columns:
rec,Comment,gpt_label

example:
91293,I am celiac - meals were good.  P.S. I was born at this hospital,disability-status

rec: unique record identifier
Comment: patient comment text
gpt_label: comma separated labels assigned by GPT model

The goal is to convert this into a jsonl file suitable for supervised fine-tuning (SFT) of a language model.



output file: sdoh_sft_gpt.jsonl

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

input_file = "gpt_sdoh_label_5000.csv"
output_file = "sdoh_gpt_sft.jsonl"

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



def normalize_labels(label_str: str) -> str:
    """
    Normalize the gpt_label field:
    - Handle NaN/empty as 'no_sdoh'
    - Split on commas
    - Strip whitespace
    - Filter to allowed labels
    - Remove duplicates while preserving order
    """
    if pd.isna(label_str):
        return "no_sdoh"

    raw = str(label_str).strip().lower()
    if not raw:
        return "no_sdoh"

    labels = [l.strip() for l in raw.split(",") if l.strip()]

    # filter to allowed labels
    allowed = set(LABEL_COLUMNS)
    labels = [l for l in labels if l in allowed]

    if not labels:
        labels = ["no_sdoh"]

    # deduplicate preserving order
    seen = set()
    uniq = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            uniq.append(l)

    return ",".join(uniq)


def main():
    df = pd.read_csv(input_file)

    with open(output_file, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            comment = str(row["Comment"]).strip()
            labels = normalize_labels(row["gpt_label"])

            record = {
                "instruction": INSTRUCTION,
                "input": comment,
                "output": labels,
            }

            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    main()