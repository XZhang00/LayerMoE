import torch
import os, json
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import time
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

LANGUAGE_CODE = {
    "en": "eng",
    "zh": "zho_simpl",
    "es": "spa",
    "el": "ell",
    "hu": "hun",
    "tr": "tur"
}

LANGUAGE_NAME = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "el": "Greek",
    "hu": "Hungarian",
    "tr": "Turkish"
}


def load_model_and_tokenizer(args):
    model_name_or_path = args.model_path
    if not args.moe_type:
        print("base-model eval ing...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.padding_side = 'left'
        # tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = 'left'
        moe_config = PeftConfig.from_pretrained(args.adapter_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="cuda:0")
        print("$$$-moe_config", moe_config)
        model = PeftModel.from_pretrained(model, 
                                            model_id=args.adapter_path, 
                                            config=moe_config, 
                                            is_trainable=True
                                            )
    model = model.eval()
    model.cuda()
    # print(model)
    return model, tokenizer


class KeyWordOne_StoppingCriteria(StoppingCriteria):
    def __init__(self, keyword, tokenizer, device):
        self.keyword = tokenizer.encode(keyword,add_special_tokens = False,return_tensors = 'pt').squeeze().to(device)
    def __call__(self, input_ids, scores, **kwards):
        # print("@@@-1",input_ids[0])
        # print("@@@-2", self.keyword)
        if len(input_ids[0]) < len(self.keyword):
            return False
        if input_ids[0][len(input_ids[0]) - len(self.keyword):].equal(self.keyword):
            return True
        else:
            return False


def translate(src, template_str, few_shot_prefix="", stopcrieria=None):
    text = few_shot_prefix + template_str.replace("<input_src>", src).replace("<input_tgt>", "")
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    if stopcrieria is not None:
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            num_beams=6,
            early_stopping=True,
            stopping_criteria=StoppingCriteriaList([stopcrieria])
        )
    else:
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
            num_beams=6,
            early_stopping=True
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # print("@@@-id", generated_ids)
    # print(tokenizer(["\nTranslation: "]))
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("@@@-0", response)
    
    return response.split("\n")[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="your_path/Models/meta-llama/Llama-3.2-3B")

    parser.add_argument('--adapter_path', type=str, default="your_path/outputs-llama3.2-3B/G1-6B-my-stage2-no_cls")

    parser.add_argument("--moe_type", action='store_true', default=False)

    parser.add_argument("--save_path", type=str, default="your_path/logs-eval-llama3.2-3B/Llama-3.2-3B")

    parser.add_argument('--test_langs', type=str, default="en-zh")

    parser.add_argument('--test_path', type=str, default="flores101_dataset/devtest")
    parser.add_argument('--dev_path', type=str, default="flores101_dataset/dev")
    parser.add_argument("--few_shot_num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)


    parser.add_argument('--template_str',default="Translation: [<srclang>]: <input_src> [<tgtlang>]: <input_tgt>")


    args = parser.parse_args()
    print(args)

    model, tokenizer = load_model_and_tokenizer(args)

    few_shotlist = []
    random.seed(args.seed)

    print(args.model_path, "seed:", args.seed)
    test_langs = args.test_langs.split("&")
    for i_test_langs in test_langs:
        srclang, tgtlang = i_test_langs.split("-")
        print(f"@@@-src_lang: {srclang}, tgt_lang: {tgtlang}")
        dev_src_data = []
        with open(args.dev_path + f"/{LANGUAGE_CODE[srclang]}.dev", "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                tmp = line.rstrip("\n")
                if tmp != "":
                    dev_src_data.append(tmp)
        fr.close()

        dev_tgt_data = []
        with open(args.dev_path + f"/{LANGUAGE_CODE[tgtlang]}.dev", "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                tmp = line.rstrip("\n")
                if tmp != "":
                    dev_tgt_data.append(tmp)
        fr.close()

        assert len(dev_src_data) == len(dev_tgt_data)

        if few_shotlist == []:
            few_shotlist = random.sample(list(range(len(dev_src_data))), args.few_shot_num)

        template_str = args.template_str.replace("[<srclang>]", LANGUAGE_NAME[srclang]).replace("[<tgtlang>]", LANGUAGE_NAME[tgtlang])
        print(template_str)
        print("few_shot_examples_idx:", few_shotlist)

        prefix = ""
        for i_id in few_shotlist:
            prefix += template_str.replace("<input_src>", dev_src_data[i_id]).replace("<input_tgt>", dev_tgt_data[i_id]) + "\n"
        
        print(prefix)

        test_src_data = []
        with open(args.test_path + f"/{LANGUAGE_CODE[srclang]}.devtest", "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                tmp = line.rstrip("\n")
                if tmp != "":
                    test_src_data.append(tmp)
        fr.close()

        test_tgt_data = []
        with open(args.test_path + f"/{LANGUAGE_CODE[tgtlang]}.devtest", "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                tmp = line.rstrip("\n")
                if tmp != "":
                    test_tgt_data.append(tmp)
        fr.close()

        assert len(test_tgt_data) == len(test_src_data)

        st = time.time()
        test_gen_translations = []
        stopcrieria = KeyWordOne_StoppingCriteria("Translation:", tokenizer=tokenizer, device="cuda")
        for i_test_item in tqdm(test_src_data):
            cur_res = translate(i_test_item, template_str, few_shot_prefix=prefix, stopcrieria=stopcrieria)
            # print("@@@-1", len(cur_res), cur_res)
            if "Translation: " in cur_res:
                cur_res = cur_res.split("Translation: ")[0]
            # print("@@@-2", len(cur_res), cur_res)
            test_gen_translations.append(cur_res)

        et = time.time()
        print(f"cost: {et-st} s.")
        print("\n@-\t".join(test_gen_translations))

        assert len(test_gen_translations) == len(test_tgt_data) == len(test_src_data)

        save_data = []
        for i_src, i_mt, i_tgt in zip(test_src_data, test_gen_translations, test_tgt_data):
            cur_dict = {
                "src": i_src,
                "mt": i_mt,
                "ref": i_tgt
            }
            save_data.append(cur_dict)

        save_path = args.save_path + f"/{srclang}-to-{tgtlang}-seed_{str(args.seed)}.json"
        json.dump(save_data, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)



