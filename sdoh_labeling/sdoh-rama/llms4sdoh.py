import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import pathlib

model_name = "YBXL/SDoH-llama-L1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

system_prompt = """
### Instruction:
Given a sentence output all SDoH factors that can be inferred from that sentence from
the following list: AdverseChildhood, Alcohol, BirthSex, Drug, EducationLevel,
EmploymentStatus, EnvironExposure, FinancialIssues, FoodInsecurity, GenderIdentity,
Insurance, Isolation, LivingStatus, LocationBornRaised, MaritalStatus, PhysicalActivity,
PhysSexAbuse, Race, SexualOrientation, Smoking, and SocialSupport.
If the sentence does not mention any SDoH factor then output - nonSDoH.
"""

# SDoH categories for parsing
sdoh_categories = [
    "AdverseChildhood", "Alcohol", "BirthSex", "Drug", "EducationLevel",
    "EmploymentStatus", "EnvironExposure", "FinancialIssues", "FoodInsecurity",
    "GenderIdentity", "Insurance", "Isolation", "LivingStatus", "LocationBornRaised",
    "MaritalStatus", "PhysicalActivity", "PhysSexAbuse", "Race", "SexualOrientation",
    "Smoking", "SocialSupport", "nonSDoH"
]

def parse_sdoh_response(response_text):
    """Parse the model's text response to extract SDoH categories"""
    response_text = response_text.strip().lower()

    found_categories = []
    for category in sdoh_categories:
        if category.lower() in response_text:
            found_categories.append(category)

    if "nonsdoh" in response_text or not found_categories:
        return ["nonSDoH"]

    # Remove nonSDoH if other categories are found
    found_categories = [cat for cat in found_categories if cat.lower() != "nonsdoh"]

    return found_categories if found_categories else ["nonSDoH"]



csv_path = pathlib.Path(
    "/home/minhle/projects/aip-btaati/minhle/Test/"
    "2023-08-01 - Pivot of comments by valence - MBC_Comments.csv"
)
date_cols = ["Experience Date", "Date of first reading"]

df_final = pd.read_csv(csv_path)                     # replace the old df_final
for col in date_cols:
    df_final[col] = pd.to_datetime(
        df_final[col].astype(str).str.strip(),
        format="mixed",
        errors="coerce"
    )

print("Total comments:", len(df_final["Comment"]))
save_every = 5000
batch_size = 8
output_preds = []

texts = df_final["Comment"].astype(str).tolist()
ids = df_final["rec"].tolist()  


pred_file = open("model_predictions_log.txt", "w", encoding="utf-8")
pred_file.write('REC\tTEXT\tPREDICTED_SDOH\tRAW_RESPONSE\n') 


# Clear GPU cache before starting
torch.cuda.empty_cache()
for i in tqdm(range(0, len(texts), batch_size), desc="Classifying batched responses"):
    try:
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        prompts = []
        for text in batch_texts:
            prompt = f"{system_prompt}\n\n### Input:\n{text}\n\n### Response:\n"
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # temperature=0.1,
                repetition_penalty=1.1
            )

            # Decode only the new tokens (response part)
            input_length = inputs.input_ids.shape[1]
            response_tokens = outputs[:, input_length:]
            responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)

        for j, (rec_id, text, response) in enumerate(zip(batch_ids,batch_texts, responses)):
            sdoh_categories_found = parse_sdoh_response(response)

            pred_label = ";".join(sdoh_categories_found)

            output_preds.append(pred_label)
            pred_file.write(f"{rec_id}\t{text.strip()}\t{pred_label}\t{response.strip()}\n")

        del inputs, outputs, response_tokens
        torch.cuda.empty_cache()

        # Debug: Print some examples from the first batch
        if i == 0:
            print("\n=== Sample Responses ===")
            for j in range(min(3, len(responses))):
                print(f"Input: {batch_texts[j][:100]}...")
                print(f"Raw Response: {responses[j]}")
                print(f"Parsed SDoH: {output_preds[j]}")
                print("-" * 50)

        if (i + batch_size) % save_every < batch_size or (i + batch_size) >= len(texts):
            df_partial = df_final.iloc[:len(output_preds)].copy()
            df_partial['LLama_SDoH_Labels'] = output_preds
            filename = f"sdoH_partial_output_{len(output_preds)}.csv"
            df_partial.to_csv(filename, index=False)
            print(f" Saved partial results to {filename}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f" Out of memory at batch {i}. Try reducing batch_size to 4 or 2.")
            torch.cuda.empty_cache()
            break
        else:
            print(f" Error at batch {i}: {e}")
            break
