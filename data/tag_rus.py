import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 配置
TOKENIZER_PATH = "/home/s2678328/tokenizers/tokenzier_Russain/auto_tokenizer_auto"
INPUT_DIR = "/home/s2678328/babylm_shuffle_nondeterministic_100M_seed41"
OUTPUT_DIR = os.path.join(INPUT_DIR, "tokenized")
BATCH_SIZE = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]

for file in input_files:
    input_path = os.path.join(INPUT_DIR, file)
    with open(input_path, "r", encoding="utf-8") as fin:
        lines = [json.loads(line)["text"].strip() for line in fin if line.strip()]

    batches = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch_text = "\n".join(lines[i:i+BATCH_SIZE])
        # 分词
        tokens = tokenizer(batch_text, add_special_tokens=False)
        batches.append({
            "text":  tokens["input_ids"]
        })

    output_path = os.path.join(OUTPUT_DIR, file.replace(".jsonl", "_tokenized.jsonl"))
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in batches:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{file} 已处理并保存到 {output_path}")

print("全部处理完成！")