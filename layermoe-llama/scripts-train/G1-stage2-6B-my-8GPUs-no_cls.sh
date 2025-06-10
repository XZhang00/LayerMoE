export PYTHONPATH=your_path/LayerMoE/LLaMA-Factory
export HF_HOME=your_path/.cache
ROOT_DIR=your_path/LayerMoE/LLaMA-Factory
LLaMA_DIR=your_path/layermoe-llama
MODEL_PATH=your_path/Models/meta-llama/Llama-3.2-3B
OUTPUT_DIR=$LLaMA_DIR/outputs-llama3.2-3B/G1-6B-layermoe-stage2-no_cls
LOG_PATH=$LLaMA_DIR/logs-llama3.2-3B/train-stage2-G1-6B-layermoe-no_cls.log



if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
chmod 777 -R $OUTPUT_DIR

STAGE1_PATH=your_path/layermoe-llama/outputs-llama3.2-3B/G1-6B-layermoe
DATASET=el2b,hu2b,tr2b,en50k,zh50k,es50k


export WANDB_DISABLED=true
/opt/anaconda3/envs/layermoe-llama/bin/deepspeed --num_gpus 8 --master_port=9902 $ROOT_DIR/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $STAGE1_PATH \
    --finetuning_type moe \
    --lpr_loss_coef 0.1 \
    --train_only_router \
    --do_train \
    --dataset_dir $ROOT_DIR/data-6B \
    --dataset $DATASET \
    --max_samples 100000 \
    --generate_lang_mask \
    --preprocessing_num_workers 128 \
    --cutoff_len 512 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    >> ${LOG_PATH} 2>&1


export HF_HOME="your_path/.cache"
ROOT_DIR=your_path/layermoe-llama


OUTPUT_PATH=your_path/layermoe-llama/outputs/eval_res
LOG_PATH=$ROOT_DIR/logs-eval-llama3.2-3B

BASE_MODEL_PATH=your_path/Models/meta-llama/Llama-3.2-3B

CUR_LOG=$LOG_PATH/G1-6B-layermoe-stage2-no_cls

PEFT_MODEL_PATH=your_path/layermoe-llama/outputs-llama3.2-3B/G1-6B-layermoe-stage2-no_cls



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
        --batch_size 2 \
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
        --batch_size 4 \
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
        --batch_size auto:4 \
        >> ${CUR_LOG}/arc-G1.log 2>&1 &


export CUDA_VISIBLE_DEVICES=7
nohup /opt/anaconda3/envs/layermoe-llama/bin/lm_eval --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks belebele_zho_Hans,belebele_ell_Grek,belebele_hun_Latn,belebele_tur_Latn,belebele_spa_Latn,belebele_eng_Latn \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 2 \
        >> ${CUR_LOG}/belebele-G1.log 2>&1 &

