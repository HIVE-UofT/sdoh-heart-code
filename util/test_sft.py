from pathlib import Path
import torch, pandas as pd, numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm

CSV_PATH = Path("/home/minhle/projects/aip-btaati/minhle/Test/2023-08-01 - Pivot of comments by valence - MBC_Comments.csv")

CHECKPOINT_ROOT = "/home/minhle/projects/aip-btaati/minhle/Test/ckpts/sft/checkpoint-702"

DAPT_ADAPTER_PATH = CHECKPOINT_ROOT + "/dapt"    # frozen, optional
SFT_ADAPTER_PATH = CHECKPOINT_ROOT + "/sft"     # ***new adapter***

# OUT_PATH            = "/home/minhle/projects/aip-btaati/minhle/sft_llama3_2_3B_heart_labelled_comments.csv"
OUT_PATH = "/home/minhle/projects/aip-btaati/minhle/sft_final_heart_labelled_comments.csv"
# N_ROWS              = 1_000                       # for a quick test
# DATE_COLS           = ["Experience Date", "Date of first reading"]
# # ╰──────────────────────────────────────────────────────────────────────────╯

# # ─────────────────────────────── Data prep
# df = pd.read_csv(CSV_PATH)
# for col in DATE_COLS:
#     df[col] = pd.to_datetime(df[col].astype(str).str.strip(),
#                              format="mixed", errors="coerce")
# # df = df.head(N_ROWS).copy()

# # ─────────────────────────────── Model + adapters
# BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# bnb_cfg = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# # 1) Base model
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_ID,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_cfg,
# )

# # 2) Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_ROOT, use_fast=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# pad_id = tokenizer.pad_token_id
# base_model.config.pad_token_id = pad_id

# # 3) Restore adapters – first DAPT (frozen), then SFT (active)
# model = PeftModel.from_pretrained(
#     base_model,
#     DAPT_ADAPTER_PATH,
#     adapter_name="dapt",          # keep the name explicit
#     is_trainable=False,           # frozen
# )
# model.load_adapter(
#     SFT_ADAPTER_PATH,
#     adapter_name="sft",
#     is_trainable=False            # inference only
# )

# # Use ONLY the SFT adapter (matches training).  If you want both,
# # pass ["dapt", "sft"] instead.
# model.set_adapter("sft")
# model.eval()

# # ─────────────────────────────── Prompt template
# SYSTEM_MSG = (
#     "You are a medical text‑classification assistant. "
#     'Output ONLY "Yes" or "No" (without punctuation or extra words). '
#     'A "Yes" means the author is very likely\n'
#     "* a heart / cardiac patient, OR\n"
#     "* speaking on behalf of someone currently treated for a heart condition, OR\n"
#     "* describing care received in a cardiac / coronary / cardiology unit.\n"
#     "Anything else (e.g. comments from relatives, other departments, "
#     "general hospital feedback) is “No”."
# )

# @torch.inference_mode()
# def heart_label_llama(comment: str) -> bool | float:
#     """
#     Returns True → “Yes”, False → “No”, or NaN if the comment is empty / NaN.
#     """
#     if not isinstance(comment, str) or not comment.strip():
#         return np.nan

#     messages = [
#         {"role": "system", "content": SYSTEM_MSG},
#         {
#             "role": "user",
#             "content": (
#                 f'Comment: "{comment}"\n\n'
#                 'Was this comment written by a heart patient? Answer "Yes" or "No" only.'
#             ),
#         },
#     ]
#     input_ids = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     ).to(model.device)

#     terminators = [
#         tokenizer.eos_token_id,
#         tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#     ]

#     output_ids = model.generate(
#         input_ids,
#         max_new_tokens=2,          # “Yes”/“No” fits in 1–2 tokens
#         eos_token_id=terminators,
#         do_sample=False,           # deterministic
#     )

#     answer = tokenizer.decode(
#         output_ids[0][input_ids.shape[-1]:],
#         skip_special_tokens=True
#     ).strip().lower()

#     if answer.startswith("yes"):
#         return True
#     if answer.startswith("no"):
#         return False
#     # Fallback if the model misbehaves
#     raise ValueError(f"Unexpected reply: {answer!r}")

# # ─────────────────────────────── Run labelling
# tqdm.pandas()
# df["is_heart_patient"] = df["Comment"].progress_apply(heart_label_llama)

# # ─────────────────────────────── Diagnostics & save
# n_yes = int(df["is_heart_patient"].sum())
# print(f"Number of heart patients identified: {n_yes}")

# examples = df.loc[df["is_heart_patient"] == True, "Comment"].dropna().head()
# print("\nExample comments from heart patients:")
# for i, txt in enumerate(examples, 1):
#     print(f"{i}. {txt}")

# df.to_csv(OUT_PATH, index=False)
# print(f"\nSaved labelled file to {OUT_PATH}")








# -------------------------------------------#
# start parsing from OUT_PATH anew
csv_path = OUT_PATH
date_cols = ["Experience Date", "Date of first reading"]

df = pd.read_csv(csv_path)
for col in date_cols:
    df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format="mixed", errors="coerce")

# print out some examples of comments labelled as heart patients
heart_comments = df[df["is_heart_patient"] == True]["Comment"].dropna()
print(f"Number of heart patients identified: {len(heart_comments)}")
print("\nExample comments from heart patients:")
# print 5 random comments
for i, comment in enumerate(heart_comments.sample(5), 1):
    print(f"{i}. {comment}")

