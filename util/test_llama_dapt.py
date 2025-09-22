import os
from pathlib import Path
import torch, pandas as pd, numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel  
from tqdm import tqdm

csv_path = Path("/home/minhle/projects/aip-btaati/minhle/Test/2023-08-01 - Pivot of comments by valence - MBC_Comments.csv")
date_cols = ["Experience Date", "Date of first reading"]

df = pd.read_csv(csv_path)
for col in date_cols:
    df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format="mixed", errors="coerce")
df = df.head(1000).copy()


BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH   = "/home/minhle/projects/aip-btaati/minhle/Test/ckpts/dapt/checkpoint-10000"   # folder that holds adapter_model.safetensors
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id


bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_cfg,
)
base_model.config.pad_token_id = pad_id   # keeps generation happy

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()    

# ------------------------------------------------------------------
# 3 – Prompt template
SYSTEM_MSG = (
    "You are a medical text-classification assistant. "
    'Output ONLY "Yes" or "No" (without punctuation or extra words). '
    'A "Yes" means the author is very likely  '
    "* a heart / cardiac patient, OR "
    "* speaking on behalf of someone currently treated for a heart condition, OR "
    "* describing care received in a cardiac / coronary / cardiology unit. "
    "Anything else (e.g. comments from relatives, other departments, general hospital feedback) is “No”."
)

@torch.inference_mode()
def heart_label_llama(comment: str) -> bool | float:
    if not isinstance(comment, str) or not comment.strip():
        return np.nan
    # Chat messages -> model prompt (Hugging Face chat template handles the <|eot_id|> tokens)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": (
                f'Comment: "{comment}"\n\n'
                'Was this comment written by a heart patient? Answer "Yes" or "No" only.'
            ),
        },
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=2,         # “Yes”/“No” fits in 1–2 tokens
        eos_token_id=terminators,
        do_sample=False,  # deterministic output
    )

    answer = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()
    # Robust fallback in case the model adds extra text
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    raise ValueError(f"Unexpected reply: {answer!r}")

# ------------------------------------------------------------------
# 4 – Run classification with a progress bar
tqdm.pandas()
df["is_heart_patient"] = df["Comment"].progress_apply(heart_label_llama)

# ------------------------------------------------------------------
# 5 – Diagnostics and save
print(f"Number of heart patients identified: {df['is_heart_patient'].sum()}")

examples = df[df["is_heart_patient"] == True]["Comment"].dropna().head()
print("\nExample comments from heart patients:")
for i, txt in enumerate(examples, 1):
    print(f"{i}. {txt}")

out_path = Path("/home/minhle/projects/aip-btaati/minhle/dapt_llama3_2_3B_heart_labelled_comments.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved labelled file to {out_path}")

# print out some examples of comments labelled as heart patients
heart_comments = df[df["is_heart_patient"] == True]["Comment"].dropna()
print("\nExample comments from heart patients:")
# print 5 comments
for i, txt in enumerate(heart_comments.head(5), 1):
    print(f"{i}. {txt}")