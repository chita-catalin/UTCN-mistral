![alt text](img.png)

Project story — from raw meters to (attempted) model

This repo started as a simple idea: ask natural-language questions about the LCL smart-meter dataset and get back MongoDB queries (and answers).

1) Research & data

We based the work on the Low Carbon London (LCL) smart meter study (2011–2014), which contains half-hourly consumption with basic metadata.

We normalized the records into a single MongoDB collection energy.readings with this shape:

{
  _id: ObjectId('68b77...'),
  meter_id: 'MAC000002',
  ts: ISODate('2012-10-12T08:30:00.000Z'),
  day: '2012-10-12',
  kwh: 0,
  tariff_type: 'Std',
  tariff_label: null
}


A tiny Node script verified ingestion and basic queries (find one doc; first 5 for a meter).

2) Test data for the model

Goal: teach a model to turn plain English ↦ Mongo filter/aggregation JSON (and optionally a post-process hint).

We generated NL ↔ Mongo pairs covering:

by‐meter and by‐day filters,

ranges (date & kWh),

“latest N”,

simple aggregations (daily/weekly/monthly totals & averages),

tariff filters.

To keep memory usage low, generation appended one JSONL example at a time and we saved separate train/val/test splits.

Format we used (chat style for instruction-tuned models):

{"messages":[
  {"role":"system","content":"You translate user questions about energy readings into MongoDB JSON. Return only JSON."},
  {"role":"user","content":"Show the latest 5 readings for MAC000008."},
  {"role":"assistant","content":"{\"mongo\":{\"find\":\"readings\",\"filter\":{\"meter_id\":\"MAC000008\"},\"sort\":{\"ts\":-1},\"limit\":5},\"postprocess\":null}"}
]}

3) First training attempt — local machine

Target: Mistral-7B-Instruct with QLoRA (4-bit) so we could keep the base model frozen.

Reality on the host (Intel CPU, no NVIDIA GPU):

Hugging Face auth & gated model access: fixed by logging in.

Tokenizer deps:

Cannot instantiate tokenizer … install sentencepiece / protobuf


→ fixed by installing sentencepiece and protobuf.

TRL/Transformers version skew (older stack):

SFTConfig(... max_seq_length=) unexpected
TrainingArguments(... evaluation_strategy=) unexpected
SFTTrainer(... tokenizer=) unexpected


→ we refactored down to a minimal HF Trainer + PEFT LoRA script.

Show-stopper: QLoRA 4-bit relies on bitsandbytes + CUDA. On CPU:

CUDA is required but not available for bitsandbytes


We switched to a small base (TinyLlama-1.1B) and confirmed CPU LoRA training works, but 7B on CPU is not practical.

Takeaway: without a GPU, 7B QLoRA isn’t viable locally. Use a smaller model or rent a GPU.

4) Second training attempt — Hugging Face AutoTrain

We moved to AutoTrain for a click-and-go experience.

Prep for AutoTrain

AutoTrain’s “LLM SFT (Basic)” expects a single text field, so we converted our messages examples using the Mistral chat template:

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


…producing train_text.jsonl and val_text.jsonl with one { "text": "..." } per line.

What happened

Run 1 (self-hosted UI, Local):
AutoTrain tried to quantize (quantization: int4) on CPU → bnb CUDA error.

Run 2 (self-hosted UI, Space/T4):
We pointed to a T4, but an environment variable in the Space was wrong:

OMP_NUM_THREADS='7500m' → ValueError: invalid literal for int() with base 10


Fixed by setting

OMP_NUM_THREADS=8
MKL_NUM_THREADS=8


and restarting the Space.

Also had to ensure:

Hardware = Space (not Local),

distributed_backend = none (single GPU),

mixed_precision = fp16 (T4 doesn’t do bf16),

quantization = int4 (bnb) ON GPU only,

uploaded both train & val files.

Even with those fixes, we still hit “instant success” → “idle” when misconfigurations slipped in (e.g., missing val split, wrong base model cached as openai/gpt-oss-20b, or not restarting after env changes). Net result so far: training hasn’t completed on AutoTrain yet.

5) Where things stand

✅ Data researched (LCL), cleaned, and loaded into MongoDB
✅ Node access verified
✅ Synthetic train/val/test NL↔Mongo datasets created
✅ CPU LoRA path validated on a small model (TinyLlama)
❌ Mistral-7B QLoRA: blocked locally (no CUDA)
❌ AutoTrain (T4): job setup issues (env var / hardware mode / quantization) prevented a clean run

6) Lessons learned

Environment > everything. Most “model problems” were environment/config: CUDA vs CPU, quantization backend, library versions, and even a single bad env var.

Match quantization to hardware. bitsandbytes (int4/int8) requires CUDA. On CPU, use no quantization (and a smaller base).

Keep TRL/Transformers aligned or fall back to a plain HF Trainer + PEFT to dodge API drift.

Start small, then scale. Prove the pipeline on a 1–3B model locally; then move to 7B on a GPU.

7) Next steps

Finish the T4 run on AutoTrain:

Hardware = Space, not Local

Base = mistralai/Mistral-7B-Instruct-v0.2 (or v0.3 if accessible)

JSON params (single-GPU friendly):

{
  "trainer": "sft",
  "block_size": 512,
  "epochs": 1,
  "batch_size": 1,
  "gradient_accumulation": 16,
  "lr": 0.0002,
  "mixed_precision": "fp16",
  "optimizer": "adamw_torch",
  "scheduler": "linear",
  "peft": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "quantization": "int4",
  "distributed_backend": "none",
  "chat_template": "none",
  "padding": "right",
  "push_to_hub": true
}


If AutoTrain is unavailable, run the same recipe in Colab (T4/A10/L4) with the QLoRA script.

Plug the trained adapter into the agent loop:

model outputs a Mongo JSON plan,

Node executes against MongoDB,

agent feeds result rows back to the model for a final answer.

8) What to expect once it trains

Given our dataset is focused and the target task is NL → Mongo JSON, a small LoRA on Mistral-7B should:

generalize across meters/dates/aggregations,

stay in-policy (JSON-only),

and be robust to wording variants (“most recent”, “latest”, “past week”, etc.).