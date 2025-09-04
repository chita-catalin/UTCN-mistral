#!/usr/bin/env python
# Minimal, TRL-compatible CPU LoRA trainer for chat-style JSONL

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def fmt_fn_builder(tokenizer):
    def _fmt(batch):
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
        }
    return _fmt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", default="outputs/cpu-lora")
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=128)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=256)
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
    print("Device: cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast_tokenizer, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length  # enforce truncation for old TRL

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds   = load_dataset("json", data_files=args.val_file,   split="train")

    fmt = fmt_fn_builder(tokenizer)
    train_ds = train_ds.map(fmt, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(fmt,   batched=True, remove_columns=val_ds.column_names)

    # Minimal TrainingArguments: only universally supported fields
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        report_to=[],    # avoid wandb
        fp16=False, bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,           # will be used by .evaluate() we call after training
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        # do NOT pass "packing" if your TRL is too old
    )

    trainer.train()
    # manual eval to avoid needing evaluation_strategy in TrainingArguments
    try:
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)
    except Exception as e:
        print("[WARN] Evaluation skipped:", e)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapter to:", args.output_dir)

if __name__ == "__main__":
    main()
