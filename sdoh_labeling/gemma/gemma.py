#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run zero-shot SDoH classification with Gemma Instruct.
"""

import argparse
import os
import re
import json
import difflib
import pandas as pd
from tqdm import tqdm
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

SDOH_LABELS = [
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
    "no_sdoh",
]

SYN_MAP = {
    "financial constraint": "financial-constraint",
    "financial": "financial-constraint",
    "finance": "financial-constraint",
    "living condition": "living-condition",
    "housing": "living-condition",
    "transport": "transportation",
    "education level": "education",
    "employment": "employment-status",
    "job status": "employment-status",
    "social connection": "social-connection",
    "violence": "violence-abuse",
    "abuse": "violence-abuse",
    "disability": "disability-status",
    "exercise": "physical-activity",
    "physical activity": "physical-activity",
    "substance use": "substance-use",
    "alcohol": "substance-use",
    "smoking": "substance-use",
    "sexual activity": "sexual-activity",
}

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_/]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    return s

def coerce_to_label(text: str) -> str:
    """Map model text to one of SDOH_LABELS."""
    if not isinstance(text, str):
        return "no_sdoh"
    s = text.strip()
    if not s:
        return "no_sdoh"
    parts = s.splitlines()
    if not parts:
        return "no_sdoh"
    t = parts[0]
    t = t.replace("label", "").replace("category", "").replace(":", " ").strip()
    t_norm_raw = norm(t)

    if t_norm_raw in SDOH_LABELS:
        return t_norm_raw

    t_space = re.sub("-", " ", t_norm_raw)
    if t_space in SYN_MAP:
        return SYN_MAP[t_space]
    if t_norm_raw in SYN_MAP:
        return SYN_MAP[t_norm_raw]

    for lab in SDOH_LABELS:
        if lab != "no_sdoh" and (lab in t_norm_raw or lab.replace("-", " ") in t_space):
            return lab

    close = difflib.get_close_matches(t_norm_raw, SDOH_LABELS, n=1, cutoff=0.8)
    if close:
        return close[0]

    return "no_sdoh"


def choose_dtype():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # A100/H100 etc. -> bf16
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32

def build_messages(comment: str) -> list[dict]:
    allowed = ", ".join(SDOH_LABELS)
    sys_txt = (
        "You are a strict multi-label classifier for Social Determinants of Health. "
        "Return one or more labels from the allowed list, with no extra words. "
        "If multiple apply, output a comma-separated list in any order."
    )
    usr_txt = (
        f"Allowed labels: {allowed}\n\n"
        "Task: Classify the patient comment into all applicable labels from above.\n"
        'If none apply, return "no_sdoh".\n\n'
        f"Patient comment:\n{comment}\n\n"
        "Answer with only the label or a comma-separated list of labels."
    )
    # Gemma template doesn’t support system; fold it into a single user turn.
    return [{"role": "user", "content": sys_txt + "\n\n" + usr_txt}]


def encode_batch(tokenizer, messages_list, max_length: int = 512):
    # Use chat template when available
    if getattr(tokenizer, "apply_chat_template", None):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
    else:
        # Fallback prompt (rare for these models)
        texts = []
        for msgs in messages_list:
            sys = msgs[0]["content"]
            usr = msgs[1]["content"]
            texts.append(f"### System:\n{sys}\n\n### User:\n{usr}\n\n### Assistant:\n")

    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    return end


@torch.inference_mode()
def generate_labels(model, tokenizer, comments, batch_size: int, max_new_tokens: int, device):
    results = []
    for i in tqdm(range(0, len(comments), batch_size), desc="Batches", leave=False):
        batch = comments[i : i + batch_size]
        msgs = [build_messages(c) for c in batch]
        enc = encode_batch(tokenizer, msgs).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        # slice off the prompt
        for j in range(out.size(0)):
            input_len = int(enc["attention_mask"][j].sum().item())
            gen_ids = out[j, input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(coerce_to_label(text))
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV")
    ap.add_argument("--output", required=True, help="Output CSV (resumable)")
    ap.add_argument("--text-col", default="Comment")
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    # if args.text_col not in df.columns:
    #     # try a few common fallbacks
    #     for alt in ["comment", "text", "cleaned_comment"]:
    #         if alt in df.columns:
    #             args.text_col = alt
    #             break
    df = df.dropna(subset=[args.text_col]).reset_index(drop=True)
    # get first 10 comments
    # df = df.iloc[:10]

    # Prepare/resume output
    label_col = "gemma_label"
    if os.path.exists(args.output):
        out_df = pd.read_csv(args.output)
        # align shapes/columns if schema changed
        for col in df.columns:
            if col not in out_df.columns:
                out_df[col] = df[col]
        if label_col not in out_df.columns:
            out_df[label_col] = ""
        out_df = out_df.iloc[: len(df)].copy()
    else:
        out_df = df.copy()
        out_df[label_col] = ""

    # Model
    dtype = choose_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    device = model.device

    # Process in chunks & resume
    n = len(df)
    chunk_size = args.chunk_size
    start_idx = (out_df[label_col] != "").idxmin() if (out_df[label_col] == "").any() else n
    # Ensure we don't redo completed rows
    for start in range(start_idx - (start_idx % chunk_size), n, chunk_size):
        end = min(start + chunk_size, n)
        # skip fully done chunk
        if out_df.loc[start:end-1, label_col].ne("").all():
            continue

        comments = df.loc[start:end-1, args.text_col].tolist()
        labels = generate_labels(model, tokenizer, comments, args.batch_size, args.max_new_tokens, device)
        out_df.loc[start:end-1, label_col] = labels
        out_df.to_csv(args.output, index=False)
        print(f"[gemma] Saved rows {start}..{end-1} -> {args.output}")

    # Final save
    out_df.to_csv(args.output, index=False)
    print(f"[gemma] Done. Wrote: {args.output}")

if __name__ == "__main__":
    input_path = "/home/minhle/projects/aip-btaati/minhle/Test/2023-08-01 - Pivot of comments by valence - MBC_Comments.csv"
    output_path = "/home/minhle/projects/aip-btaati/minhle/Test/gemma/test_gemma.csv"
    text_col = "Comment"
    model = "google/gemma-2-2b-it"
    batch_size = 16
    chunk_size = 5000
    max_new_tokens = 16
    sys.argv = [
        "",
        "--input", input_path,
        "--output", output_path,
        "--text-col", text_col,
        "--model", model,
        "--batch-size", str(batch_size),
        "--chunk-size", str(chunk_size),
        "--max-new-tokens", str(max_new_tokens),
    ]
    main()
