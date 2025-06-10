
from datasets import load_dataset
import json

path = "your_path/Datasets/cerebras/SlimPajama-627B"
src = load_dataset(path)['train'].shuffle(seed=22)

print(len(src))

res = []
for k in src:
    res.append(k)
    if len(res) == 50000:
        break

with open("your_path/Datasets/cerebras/SlimPajama-627B/SlimPajama-50K.jsonl", 'w', encoding="utf-8") as f:
    for k in res:
        f.write(json.dumps(k) + '\n')