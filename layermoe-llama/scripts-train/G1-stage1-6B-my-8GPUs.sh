export PYTHONPATH=your_path/LayerMoE/LLaMA-Factory
export HF_HOME=your_path/.cache
ROOT_DIR=your_path/LayerMoE/LLaMA-Factory
LLaMA_DIR=your_path/layermoe-llama
MODEL_PATH=your_path/Models/meta-llama/Llama-3.2-3B
OUTPUT_DIR=$LLaMA_DIR/outputs-llama3.2-3B/G1-6B-layermoe
LOG_PATH=$LLaMA_DIR/logs-llama3.2-3B/train-stage1-G1-6B-layermoe.log

DATASET=el2b,hu2b,tr2b
GPU_NUM=8

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
chmod 777 -R $OUTPUT_DIR

nvidia-smi >> ${LOG_PATH} 2>&1

# History_PATH=
#     --adapter_name_or_path $History_PATH \
#     --resume_from_checkpoint $History_PATH \


export WANDB_DISABLED=true
/opt/anaconda3/envs/layermoe-llama/bin/deepspeed --num_gpus ${GPU_NUM} --master_port=9902 $ROOT_DIR/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --finetuning_type moe \
    --topk 2 \
    --ada_moe_num_experts_list 5,5,5,4,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4 \
    --aux_loss_coef 0.01 \
    --do_train \
    --dataset_dir $ROOT_DIR/data-6B \
    --dataset $DATASET \
    --preprocessing_num_workers 128 \
    --cutoff_len 1024 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 2 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    >> ${LOG_PATH} 2>&1
