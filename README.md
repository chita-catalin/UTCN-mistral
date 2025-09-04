![alt text](img.png)

# NL → Mongo for LCL Smart‑Meter Data

Tiny repo to turn natural‑language questions into MongoDB JSON for the **Low Carbon London (LCL)** smart‑meter study.

## Data & shape
Single collection: `energy.readings`
```json
{ "_id": "...", "meter_id": "MAC000002", "ts": "2012-10-12T08:30:00Z", "day": "2012-10-12", "kwh": 0, "tariff_type": "Std", "tariff_label": null }
```
Small Node script verifies ingestion and simple queries.

## Training data (chat → JSON)
Covers by‑meter/day, ranges, “latest N”, daily/weekly/monthly totals & averages, tariff filters.
```json
{
  "messages": [
    {"role":"system","content":"You translate user questions about energy readings into MongoDB JSON. Return only JSON."},
    {"role":"user","content":"Show the latest 5 readings for MAC000008."},
    {"role":"assistant","content":"{\"mongo\":{\"find\":\"readings\",\"filter\":{\"meter_id\":\"MAC000008\"},\"sort\":{\"ts\":-1},\"limit\":5},\"postprocess\":null}"}
  ]
}
```

## Status
- ✅ LCL data loaded; Node access OK; NL↔Mongo datasets split
- ✅ CPU LoRA works on TinyLlama‑1.1B
- ❌ Mistral‑7B QLoRA on CPU (needs CUDA)
- ❌ AutoTrain (T4) runs failed due to env/hardware/quantization mismatches

## Issues encountered so far
- **bitsandbytes on CPU:** fails → use smaller model on CPU or a GPU for 7B
- **Deps:** install `sentencepiece` + `protobuf`
- **API drift (TRL/Transformers):** use plain HF `Trainer` + PEFT if needed
- **AutoTrain quantization:** int4 only on **GPU**
- **Spaces env:** set `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` (then restart)
- **Runtime:** Hardware = **Space**, `distributed_backend=none`, `mixed_precision=fp16` (T4)

## Minimal AutoTrain recipe (T4)
Base: `mistralai/Mistral-7B-Instruct-v0.2` (or v0.3)
```json
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
```

## Expected
- JSON‑only outputs; generalizes across meters/dates/aggs; robust to wording variants

## Wire‑up
1) Model → Mongo JSON plan  
2) Node executes on MongoDB  
3) Return rows → final answer

## Next
- Re‑run on T4 with the config above or use Colab (T4/A10/L4) for QLoRA and push the LoRA adapter
