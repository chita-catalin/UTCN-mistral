#!/usr/bin/env python
# CPU-only LoRA trainer for chat-style JSONL (TRL-version friendly)

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def build_formatting_fn(tokenizer):
    def _fmt(batch):
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    return _fmt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Pick a small base (1–3B) for CPU")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", default="outputs/cpu-lora")
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=128)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_seq_length", type=int, default=256)  # CPU-friendly
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--use_fast_tokenizer", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Device:", torch.device("cpu"))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast_tokenizer, token=args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length  # cap length for old TRL

    # Base model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    model.config.use_cache = False  # needed for grad checkpointing

    # LoRA (attention-only keeps it lighter)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Datasets
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds   = load_dataset("json", data_files=args.val_file,   split="train")

    fmt = build_formatting_fn(tokenizer)
    train_ds = train_ds.map(fmt, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(fmt,   batched=True, remove_columns=val_ds.column_names)

    # Plain TrainingArguments (works across TRL versions)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],          # no wandb
        fp16=False, bf16=False,
        gradient_checkpointing=True,
    )

    # IMPORTANT: pass max_seq_length directly to SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,   # if your TRL is too old and errors on this arg, just remove it
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapter to:", args.output_dir)

if __name__ == "__main__":
    main()
