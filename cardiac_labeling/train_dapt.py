import os, math, torch, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CORPUS_PATH = "data/dapt_corpus.txt"
OUTPUT_DIR = "ckpts/dapt"
MAX_SEQ_LEN = 2048
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 1e-4
MAX_STEPS = 10000
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj","k_proj","v_proj",
    "o_proj","gate_proj","up_proj","down_proj"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("text", data_files=CORPUS_PATH, split="train")

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])


from transformers import BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_cfg,
    device_map="auto",
)

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)
model = get_peft_model(base_model, lora_cfg)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    max_steps=MAX_STEPS,
    lr_scheduler_type="cosine",
    warmup_steps=200,
    logging_steps=50,
    save_steps=1000,
    bf16=True,
    optim="adamw_torch_fused",
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

if __name__ == "__main__":
    trainer.train()
    # save last checkpoint as "final"
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")