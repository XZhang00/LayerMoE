import json, os, jsonlines
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, torch
from tqdm import tqdm
from types import MethodType
from torch.nn import functional as F
from peft import PeftModel, PeftConfig


data_path_dict = {
    "en": "your_path/SlimPajama-50K.jsonl",
    "es": "your_path/es-qwen-50K.jsonl",
    "zh": "your_path/2023-14_zh_middle_0009-50K.jsonl",
    "el": "your_path/el-qwen-2B.jsonl",
    "hu": "your_path/hu-qwen-2B.jsonl",
    "tr": "your_path/tr-qwen-2B.jsonl",
    "bn": "your_path/bn-qwen-2B.jsonl",
    "hi": "your_path/hi-qwen-2B.jsonl",
    "ne": "your_path/ne-qwen-2B.jsonl"
}

def main(model, tokenizer, langs):
    
    num_layers = model.config.num_hidden_layers

    tokens_limit = 150000
    tokens_select = 100000

    all_langs_hidden_states = {}
    for lang in langs.split("_"):
        data_path = data_path_dict[lang]
        data_select = []
        all_tokens = 0
        with jsonlines.open(data_path) as f:
            for line in f:
                if all_tokens > tokens_limit: break
                tmp_tok_res = tokenizer(line["text"])["input_ids"]
                if len(tmp_tok_res) > 32000: continue   # 大于10000会爆显存
                all_tokens += len(tmp_tok_res)
                data_select.append(line["text"])
        f.close()
        print(f"lang: {lang}, data_num: {len(data_select)}, all_tokens: {all_tokens}")


        global hidden_states_lang
        hidden_states_lang = [[] for i in range(num_layers)]

        def factory(idx):
            def new_forward(self, x: torch.Tensor, input_ids=None) -> torch.Tensor:
                hidden_states_lang[idx].append(x.reshape(-1, x.size(-1)).cpu())
                previous_dtype = x.dtype
                router = self.moe_router_embedding[self.active_adapter[0]]  # b x s x e

                classify_res = None
                if self.group_nums is not None and self.group_nums != 0:
                    group_classify = self.moe_token_classify[self.active_adapter[0]]
                    assert group_classify is not None
                    result, router_logits, classify_res = self.classify_route_func(x, router, group_classify, self.active_adapter[0])
                else:
                    # print("&&&&-topk_route")
                    result, router_logits = self.topk_route(x, router, self.active_adapter[0])

                result = result.to(previous_dtype)
                return result, router_logits, classify_res
                
            return new_forward
        
        for i in range(num_layers):
            obj = model.base_model.model.model.layers[i].mlp
            obj.forward = MethodType(factory(i), obj)

        print("!!!hidden_states_lang[0]", len(hidden_states_lang[0]))
        for i in range(len(data_select)):
            prompt = data_select[i]
            inputs = tokenizer(prompt, max_length=10000, truncation=True, return_tensors="pt").to("cuda:0")
            # if i % 200 == 0:
            #     print(i, "@@@", inputs.input_ids.size())
            # generate_ids = model.generate(**inputs, do_sample=True, max_new_tokens=1)
            with torch.no_grad():
                # _ = model(**inputs)
      
                try:
                    _ = model(**inputs)
                except:
                    print(i, "@@@-error-@@@", inputs.input_ids.size())
        print("!!!hidden_states_lang[0]", len(hidden_states_lang[0]))
        assert len(hidden_states_lang) == num_layers
        hidden_states_lang_layers = {}
        for i_layers in range(num_layers):
            tmp = torch.cat(hidden_states_lang[i_layers], dim=0)
            assert tmp.size(0) > tokens_select
            hidden_states_lang_layers[i_layers] = tmp[:tokens_select]
            # print(hidden_states_lang_layers[i_layers].size())

        all_langs_hidden_states[lang] = hidden_states_lang_layers
    
    dif_layers_cos_sim = []
    for i_layer in range(num_layers):
        # print(langs.split("_"))
        lang_a, lang_b = langs.split("_")
        tmp_a = all_langs_hidden_states[lang_a][i_layer]
        tmp_a = tmp_a.to("cuda:1")
        tmp_b = all_langs_hidden_states[lang_b][i_layer]
        tmp_b = tmp_b.to("cuda:1")
        # print(tmp_a.size(), tmp_b.size())
        assert tmp_a.size() == tmp_b.size()
        # tmp_cos = torch.cosine_similarity(tmp_a, tmp_b, dim=-1)
        tmp_a_norm = tmp_a / tmp_a.norm(dim=1, keepdim=True)
        tmp_b_norm = tmp_b / tmp_b.norm(dim=1, keepdim=True)
        tmp_cos = torch.matmul(tmp_a_norm, tmp_b_norm.T)
        # print(tmp_cos.size(), tmp_cos[0][:10])
        dif_layers_cos_sim.append(tmp_cos.mean().cpu())
        del tmp_a
        del tmp_b
        del tmp_a_norm
        del tmp_b_norm
        del tmp_cos
    
    for i_layer in range(num_layers):
        # print(f"layer-{str(i_layer)}, cos-sim: {dif_layers_cos_sim[i_layer]:.4f}")
        print(f"{dif_layers_cos_sim[i_layer]:.4f}")

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="your_path/Models/Qwen/Qwen1.5-1.8B")
    # parser.add_argument("-m", "--model", type=str, default="your_path/Models/meta-llama/Llama-3.2-3B")
    
    parser.add_argument("--new_langs", type=str, 
                        default="bn_hi_ne")
    parser.add_argument("--old_langs", type=str, 
                        default="en_es_zh_el_hu_tr") 
    parser.add_argument("--adapter_path", type=str, default="your_path/MoE-LPR-llama/outputs-llama3.2-3B/6B-layermoe-g1-stage2")
    
    args = parser.parse_args()

    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    moe_config = PeftConfig.from_pretrained(args.adapter_path)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="cuda:0")
    print("$$$-moe_config", moe_config)
    model = PeftModel.from_pretrained(model, 
                                        model_id=args.adapter_path, 
                                        config=moe_config, 
                                        is_trainable=True
                                        )
    # main(model, tokenizer, args.lang)

    new_langs = args.new_langs.split("_")
    old_langs = args.old_langs.split("_")
    
    for i in range(len(new_langs)):
        for j in range(i+1, len(new_langs)):
            cur_langs = new_langs[i] + "_" + new_langs[j]
            print("+++++", cur_langs)
            main(model, tokenizer, cur_langs)

    for i in new_langs:
        for j in old_langs:
            cur_langs = i + "_" + j
            print("+++++", cur_langs)
            main(model, tokenizer, cur_langs)