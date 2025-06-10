from transformers import AutoTokenizer
import argparse
import json, os
from tqdm import tqdm
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="hu")
    parser.add_argument("--tok", type=str, 
                        default="your_path/Models/qwen1.5-1.8b")
    return parser.parse_args()


def count(data):
    c = 0
    for line in tqdm(data):
        c += len(line['ids']) - 2
    return c


def tok(example):
    return {"ids": tokenizer(example['text'])['input_ids']}


def collect(data, tokenizer, document_nums=50000):
    res = []
    for k in tqdm(data):
        res.append(k)
        if len(res) == document_nums:
            break
    return res


if __name__ == "__main__":
    args = get_args()

    lang = args.lang
    src = load_dataset(f"your_path/Datasets/CulturaX-per30_parquet/{lang}")['train'].shuffle(seed=22)
    print(src)
    print(len(src))

    tok_model_path = args.tok
    tokenizer = AutoTokenizer.from_pretrained(tok_model_path)

    final = collect(src, tokenizer, document_nums=50000)

    save_path = "your_path/Datasets/CulturaX-per30_parquet/sample-data"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if "llama" in tok_model_path.lower():
        save_path_z = save_path + "/" + lang + "-llama"
    elif "qwen" in tok_model_path.lower():
        save_path_z = save_path + "/" + lang + "-qwen"

    with open(f"{save_path_z}-50K.jsonl", 'w', encoding="utf-8") as f:
        for k in tqdm(final):
            f.write(json.dumps(k) + '\n')

# , encoding="utf-8"