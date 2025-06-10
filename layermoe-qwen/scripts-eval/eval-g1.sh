export HF_HOME="your_path/.cache"
ROOT_DIR=your_path/LayerMoE/LLaMA-Factory


OUTPUT_PATH=your_path/LayerMoE/LLaMA-Factory/outputs/eval_res
LOG_PATH=$ROOT_DIR/logs-eval

BASE_MODEL_PATH=your_path/Models/Qwen/Qwen1.5-1.8B

TAG=cur-test-model
CUR_LOG=$LOG_PATH/$TAG
PEFT_MODEL_PATH=your_path/LayerMoE/LLaMA-Factory/outputs/$TAG


if [ ! -d "$CUR_LOG" ]; then
  mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG



export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16",parallelize=True \
        --tasks mmlu_el \
        --num_fewshot 5 \
        --seed 0,1234,1234,1234 \
        --output_path $OUTPUT_PATH \
        --batch_size 1 \
        >> ${CUR_LOG}/mmlu-el.log 2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu_tr \
        --device cuda:0 \
        --num_fewshot 4 \
        --seed 0,1234,1234,1234 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/mmlu-tr.log 2>&1 &


# You can split these tasks with more GPUs for faster evaluation.
export CUDA_VISIBLE_DEVICES=5
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu,cmmlu,m_mmlu_es,m_mmlu_hu \
        --device cuda:0 \
        --num_fewshot 5 \
        --seed 0,1234,1234,1234 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/mmlu-en_zh_es_hu.log 2>&1 &



export CUDA_VISIBLE_DEVICES=6
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_hu,hellaswag_tr,hellaswag_el,hellaswag_zh,hellaswag,hellaswag_es \
        --device cuda:0 \
        --num_fewshot 10 \
        --seed 0,1234,1234,1234 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/hellaswag.log 2>&1 &


# You can split these tasks with more GPUs for faster evaluation. e.g.,
# export CUDA_VISIBLE_DEVICES=0
# nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
#         --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
#         --tasks hellaswag_hu \
#         --device cuda:0 \
#         --num_fewshot 10 \
#         --seed 0,1234,1234,1234 \
#         --output_path $OUTPUT_PATH \
#         --batch_size 4 \
#         >> ${CUR_LOG}/hellaswag.log 2>&1 &


export CUDA_VISIBLE_DEVICES=7
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks arc_challenge,arc_es,arc_hu,arc_zh,arc_el,arc_tr \
        --device cuda:0 \
        --num_fewshot 25 \
        --seed 0,1234,1234,1234 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/arc.log 2>&1 &



wait

export CUDA_VISIBLE_DEVICES=0
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks belebele_zho_Hans,belebele_ell_Grek,belebele_hun_Latn,belebele_tur_Latn,belebele_spa_Latn,belebele_eng_Latn \
        --device cuda:0 \
        --num_fewshot 5 \
        --seed 0,1234,1234,1234 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:48 \
        >> ${CUR_LOG}/belebele.log 2>&1 &