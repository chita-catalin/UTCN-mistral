# Converts chat JSONL with {"messages":[{role,content},...]} to {"text": "..."} using Mistral template
import json, sys
from transformers import AutoTokenizer

BASE = "mistralai/Mistral-7B-Instruct-v0.3"  # v0.2 also fine
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

def convert(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            msgs = obj["messages"]
            # one concatenated training string (no generation prompt, includes assistant target)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Usage: python prep_autotrain.py data/train.jsonl data/train_text.jsonl
    in_file, out_file = sys.argv[1], sys.argv[2]
    convert(in_file, out_file)
