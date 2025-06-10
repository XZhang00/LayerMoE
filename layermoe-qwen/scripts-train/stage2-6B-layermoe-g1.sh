export PYTHONPATH=your_path/LayerMoE/LLaMA-Factory
export HF_HOME=your_path/.cache

ROOT_DIR=your_path/LayerMoE/LLaMA-Factory
MODEL_PATH=your_path/Models/Qwen/Qwen1.5-1.8B
OUTPUT_DIR=$ROOT_DIR/outputs/6B-layermoe-g1-stage2
LOG_PATH=$ROOT_DIR/logs/train-stage2-6B-layermoe-g1.log



if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
chmod 777 -R $OUTPUT_DIR

STAGE1_PATH=your_path/LayerMoE/LLaMA-Factory/outputs/6B-layermoe-g1-8GPUs
DATASET=el2b,hu2b,tr2b,en50k,zh50k,es50k


export WANDB_DISABLED=true
/opt/anaconda3/envs/layermoe-qwen/bin/deepspeed --num_gpus 8 --master_port=9905 $ROOT_DIR/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --adapter_name_or_path $STAGE1_PATH \
    --finetuning_type moe \
    --group_nums 0,0,0,0,0,2,2,0,0,0,2,2,2,2,0,0,0,0,0,0,0,0,0,2 \
    --old_moe_num_experts_list 5,4,3,5,4,3,2,3,3,2,2,2,2,2,2,2,2,3,3,4,4,4,3,3 \
    --sequential_add_mask True \
    --different_old_experts True \
    --classify_loss_coef 0.1 \
    --lpr_loss_coef 0.1 \
    --train_only_router \
    --do_train \
    --dataset_dir your_path/data-12B \
    --dataset $DATASET \
    --max_samples 100000 \
    --generate_lang_mask \
    --preprocessing_num_workers 128 \
    --cutoff_len 512 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 2 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    >> ${LOG_PATH} 2>&1



