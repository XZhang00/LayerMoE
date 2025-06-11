import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from comet import load_from_checkpoint
# from comet import load_model
from comet import load_from_checkpoint
import json
from numpy import mean

model_path = "your_path/Models/Unbabel/wmt22-comet-da/checkpoints/model.ckpt"
model = load_from_checkpoint(model_path)
print("model loaded!!!")


res_path = "your_path/logs-eval/G1-6B-base-3experts-stage2/flores"
seed = 42

test_langs = "en-zh&zh-en&en-es&es-en&en-el&el-en&en-hu&hu-en&en-tr&tr-en"
# test_langs = "en-zh&zh-en"
test_langs = test_langs.split("&")

comet_res = []
for i_test_langs in test_langs:
    srclang, tgtlang = i_test_langs.split("-")
    cur_file = res_path + f"/{srclang}-to-{tgtlang}-seed_{str(seed)}.json"
    data = json.load(open(cur_file))
    model_output = model.predict(data, batch_size=8, gpus=1)
    # print(model_output)
    comet_res.append(round(model_output["system_score"]*100, 2))

print(test_langs)
print(comet_res)

out_res = comet_res[:4] + [round(mean(comet_res[:4]), 2)] + comet_res[4:] + [round(mean(comet_res[4:]), 2)]
out_res = [str(i) for i in out_res]
print("\t".join(out_res))


