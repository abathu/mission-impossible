import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm

# 指定你的tokenizer路径
tokenizer = AutoTokenizer.from_pretrained("/home/s2678328/tokenizers/tokenzier_Russain/auto_tokenizer_auto")

# 数据目录
data_dir = "/home/s2678328/babylm_shuffle_nondeterministic_100M_seed41"
# 输出目录（可自定义）
output_dir = os.path.join(data_dir, "tokenized")
os.makedirs(output_dir, exist_ok=True)

# 遍历所有jsonl文件
for filename in os.listdir(data_dir):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jsonl", "_tokenized.jsonl"))
        print(f"Processing {input_path} ...")
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=filename):
                obj = json.loads(line)
                text = obj["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False)
                # 保存为token id序列
                fout.write(json.dumps({"input_ids": tokens, "text": text}) + "\n")

print("全部文件分词完成，输出在:", output_dir)