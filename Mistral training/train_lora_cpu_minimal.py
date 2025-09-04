#!/usr/bin/env python
# Minimal CPU-only LoRA trainer using plain Hugging Face Trainer (no TRL deps)

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def build_chat_texts(tokenizer):
    # Turn {"messages":[...]} into a single chat-formatted string
    def _fmt(batch):
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    return _fmt


def tokenize_fn(tokenizer, max_len):
    def _tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding=False,   # pad dynamically in the collator
        )
        return enc
    return _tok


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Small CPU-friendly base (1–3B), e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0")
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast_tokenizer, token=args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_length

    # Base model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA (attention-only keeps memory lower)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # Datasets → text → tokens
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds   = load_dataset("json", data_files=args.val_file,   split="train")

    fmt = build_chat_texts(tokenizer)
    train_ds = train_ds.map(fmt, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(fmt,   batched=True, remove_columns=val_ds.column_names)

    tok = tokenize_fn(tokenizer, args.max_seq_length)
    train_ds = train_ds.map(tok, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(tok,   batched=True, remove_columns=val_ds.column_names)

    # Data collator for causal LM (creates labels = input_ids and handles padding)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Ultra-compatible TrainingArguments (avoid newer/iffy fields)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,  # important when passing dicts from collator
        fp16=False, bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    # Try a quick eval (older transformers may not support some metrics; swallow errors)
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
