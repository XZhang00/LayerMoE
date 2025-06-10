#!/bin/bash

# Function to check if all GPUs have zero memory usage
check_gpus_memory_free() {
    # Get the memory usage for each GPU
    memory_usages=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # Check each GPU memory usage
    for usage in $memory_usages; do
        if [ "$usage" -ne 0 ]; then
            return 1 # At least one GPU is not free
        fi
    done

    return 0 # All GPUs are free
}

# Main loop to wait until all GPUs have zero memory usage
while true; do
    if check_gpus_memory_free; then
        echo "All GPUs have zero memory usage. Running your code..."
        # Place your code here
        # For example: python your_script.py
        break
    else
        echo "GPUs are not free. Checking again in 5 minutes..."
        sleep 300
    fi
done


export HF_HOME="your_path/.cache"
ROOT_DIR=your_path/layermoe-llama


OUTPUT_PATH=your_path/layermoe-llama/outputs/eval_res
LOG_PATH=$ROOT_DIR/logs-eval

BASE_MODEL_PATH=your_path/Models/meta-llama/Llama-3.2-1B

CUR_LOG=$LOG_PATH/G1-6B-base-3experts-stage2

PEFT_MODEL_PATH=your_path/layermoe-llama/outputs/G1-6B-base-3experts-stage2



if [ ! -d "$CUR_LOG" ]; then
  mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG




export CUDA_VISIBLE_DEVICES=0
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu_tr,m_mmlu_es,m_mmlu_hu \
        --device cuda:0 \
        --num_fewshot 4 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/mmlu-tr_es_hu.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/mmlu-en.log 2>&1 &


export CUDA_VISIBLE_DEVICES=2
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks cmmlu \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/mmlu-zh.log 2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu_el \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 1 \
        >> ${CUR_LOG}/mmlu-el.log 2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_el,hellaswag_hu,hellaswag_tr \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/hellaswag-el_hu_tr.log 2>&1 &


export CUDA_VISIBLE_DEVICES=5
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag,hellaswag_es,hellaswag_zh  \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/hellaswag-en_es_zh.log 2>&1 &


export CUDA_VISIBLE_DEVICES=6
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks arc_challenge,arc_es,arc_hu,arc_zh,arc_el,arc_tr \
        --device cuda:0 \
        --num_fewshot 25 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/arc-G1.log 2>&1 &


export CUDA_VISIBLE_DEVICES=7
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks belebele_zho_Hans,belebele_ell_Grek,belebele_hun_Latn,belebele_tur_Latn,belebele_spa_Latn,belebele_eng_Latn \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/belebele-G1.log 2>&1 &

