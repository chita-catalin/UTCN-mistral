#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU-only LoRA trainer for chat-style JSONL.
Use this when you have NO NVIDIA GPU.

Data format per line:
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"{...JSON...}"]}

Tip: start with a small model (1–3B). Examples:
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0   (Apache-2.0)
  - Qwen2.5-1.5B-Instruct                (permissive)
  - Llama-3.2-1B-Instruct                (Meta license)

Requires: transformers, datasets, peft, trl, sentencepiece, protobuf
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def build_formatting_fn(tokenizer):
    def _fmt(batch):
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    return _fmt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model repo or local path (choose 1–3B for CPU).")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--output_dir", default="outputs/cpu-lora")
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_seq_length", type=int, default=256)  # keep small on CPU
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (optional for gated models)")
    ap.add_argument("--use_fast_tokenizer", action="store_true", help="Try fast tokenizer (requires tokenizers).")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Device:", torch.device("cpu"))

    # Tokenizer (CPU: slow is fine; fast requires tokenizers & can fail without sentencepiece)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        token=args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Base model on CPU (float32). This is heavy; choose a small base (1–3B).
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        trust_remote_code=False
    )
    # Reduce peak memory a bit during training
    model.config.use_cache = False

    # LoRA config — attention-only for smaller memory/compute
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
    val_ds   = val_ds.map(fmt, batched=True,   remove_columns=val_ds.column_names)

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
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
        report_to=["none"],
        fp16=False,  # CPU doesn't do fp16
        bf16=False,  # Unless you have BF16-capable CPU + PyTorch build
        gradient_checkpointing=True,  # saves RAM at cost of time
        optim="adamw_torch",          # CPU-friendly
        packing=False                 # safer for RAM; turn on later if stable
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapter to:", args.output_dir)

if __name__ == "__main__":
    main()
