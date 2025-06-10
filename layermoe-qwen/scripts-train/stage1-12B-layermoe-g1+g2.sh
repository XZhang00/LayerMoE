export PYTHONPATH=your_path/LayerMoE/LLaMA-Factory
export HF_HOME=your_path/.cache
ROOT_DIR=your_path/LayerMoE/LLaMA-Factory
MODEL_PATH=your_path/Models/Qwen/Qwen1.5-1.8B
OUTPUT_DIR=$ROOT_DIR/outputs/12B-layermoe-g1+g2-8GPUs
LOG_PATH=$ROOT_DIR/logs/train-stage1-12B-layermoe-g1+g2-8GPUs.log

DATASET=el2b,hu2b,tr2b,bn2b,hi2b,ne2b
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
/opt/anaconda3/envs/layermoe-qwen/bin/deepspeed --num_gpus ${GPU_NUM} --master_port=9902 $ROOT_DIR/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --data_seed 42 \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --finetuning_type moe \
    --topk 2 \
    --ada_moe_num_experts_list 5,4,3,5,4,3,2,3,3,2,2,2,2,2,2,2,2,3,3,4,4,4,3,3 \
    --aux_loss_coef 0.01 \
    --do_train \
    --dataset_dir your_path/data-12B \
    --dataset $DATASET \
    --preprocessing_num_workers 128 \
    --cutoff_len 1024 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 2 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    >> ${LOG_PATH} 2>&1


