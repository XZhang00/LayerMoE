export HF_HOME="your_path/.cache"
ROOT_DIR=your_path/layermoe-llama


OUTPUT_PATH=your_path/layermoe-llama/outputs/eval_res
LOG_PATH=$ROOT_DIR/logs-eval

BASE_MODEL_PATH=your_path/Models/meta-llama/Llama-3.2-1B

CUR_LOG=$LOG_PATH/xxx

PEFT_MODEL_PATH=your_path/layermoe-llama/outputs/xxx



if [ ! -d "$CUR_LOG" ]; then
  mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG


export CUDA_VISIBLE_DEVICES=0
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks arc_challenge,arc_es,arc_zh,arc_bn,arc_hi,arc_ne \
        --device cuda:0 \
        --num_fewshot 25 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/arc-G2.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks belebele_zho_Hans,belebele_spa_Latn,belebele_eng_Latn,belebele_ben_Beng,belebele_hin_Deva,belebele_npi_Deva \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/belebele-G2.log 2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag,hellaswag_es,hellaswag_zh \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/hellaswag-en_es_zh.log 2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_bn \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/hellaswag-bn.log 2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_hi \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/hellaswag-hi.log 2>&1 &


export CUDA_VISIBLE_DEVICES=5
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_ne \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/hellaswag-ne.log 2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks cmmlu,m_mmlu_es,m_mmlu_bn,m_mmlu_hi,m_mmlu_ne \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/mmlu-es_zh_bn_hi_ne.log 2>&1 &


export CUDA_VISIBLE_DEVICES=7
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/mmlu-en.log 2>&1 &


sleep 6h