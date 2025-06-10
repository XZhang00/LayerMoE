import random

path = "your_path/Datasets/Skywork/SkyPile-150B/2023-14_zh_middle_0009.jsonl"

path_save = "your_path/Datasets/Skywork/SkyPile-150B/2023-14_zh_middle_0009-50K.jsonl"
with open(path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    random.seed(22)
    random.shuffle(lines)
    with open(path_save, 'w', encoding="utf-8") as fw:
        for line in lines[:50000]:
            fw.write(line)
        fw.close()

f.close()
