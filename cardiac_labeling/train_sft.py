import os, json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    Trainer,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DAPT_ADAPTER = "ckpts/dapt/checkpoint-10000"
SFT_OUTPUT_DIR = "ckpts/sft"
DATA_PATH = "data/heart_sft.jsonl"
MAX_SEQ_LEN = 1024
BATCH_PER_DEVICE = 4
GRAD_ACC_STEPS = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj","k_proj","v_proj",
    "o_proj","gate_proj","up_proj","down_proj"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def make_prompt(example):
    return TEMPLATE.format(**example)

raw_ds = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

def tokenize_fn(batch):
    """
    batch is a dict:
        {"instruction": [..], "input": [..], "output": [..]}
    Build one prompt per row, then tokenise.
    """
    prompts = [TEMPLATE.format(instruction=ins, input=inp) + out \
               for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"])]

    tokenised = tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LEN)
    return tokenised

ds = raw_ds.map(tokenize_fn, batched=True, remove_columns=raw_ds.column_names)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model = prepare_model_for_kbit_training(base_model)

model = PeftModel.from_pretrained(base_model, DAPT_ADAPTER, adapter_name="dapt", is_trainable=False)


sft_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model.add_adapter(adapter_name="sft", peft_config=sft_cfg)
model.set_adapter("sft")
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=SFT_OUTPUT_DIR,
    per_device_train_batch_size=BATCH_PER_DEVICE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=25,
    save_strategy="epoch",
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
    # save the *new* adapter only ( ≈ 90 MB ) plus tokenizer
    model.save_pretrained(f"{SFT_OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{SFT_OUTPUT_DIR}/final")
