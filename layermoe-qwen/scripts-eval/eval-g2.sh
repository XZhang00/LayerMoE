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


export CUDA_VISIBLE_DEVICES=0
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks arc_challenge,arc_es,arc_zh,arc_bn,arc_hi,arc_ne \
        --device cuda:0 \
        --num_fewshot 25 \
        --output_path $OUTPUT_PATH \
        --batch_size 1 \
        >> ${CUR_LOG}/arc.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks belebele_eng_Latn,belebele_spa_Latn,belebele_zho_Hans,belebele_ben_Beng,belebele_hin_Deva,belebele_npi_Deva \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:32 \
        >> ${CUR_LOG}/belebele.log 2>&1 &



export CUDA_VISIBLE_DEVICES=2
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_bn \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/hellaswag-bn.log 2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_hi \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/hellaswag-hi.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_ne \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/hellaswag-ne.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag,hellaswag_zh \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/hellaswag-en_zh.log 2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_es \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/hellaswag-es.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup /opt/anaconda3/envs/layermoe-qwen/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu,m_mmlu_es,cmmlu,m_mmlu_bn,m_mmlu_hi,m_mmlu_ne \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/mmlu.log 2>&1 &



