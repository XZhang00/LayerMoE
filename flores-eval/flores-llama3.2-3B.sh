export HF_HOME="your_path/.cache"
SAVE_PATH=your_path/LayerMoE/logs-eval-llama3.2-3B

CUR_LOG=$SAVE_PATH/Llama-3.2-3B/flores

if [ ! -d "$CUR_LOG" ]; then
  mkdir -p $CUR_LOG
fi
chmod 777 -R $CUR_LOG

export CUDA_VISIBLE_DEVICES=3
nohup /opt/anaconda3/envs/layermoe-lllama/bin/python -u flores_101_eval.py \
    --model_path your_path/Models/meta-llama/Llama-3.2-3B \
    --test_langs "en-hu&hu-en&en-tr&tr-en" \
    --save_path $CUR_LOG \
    >> $CUR_LOG/hu_tr.log 2>&1 &



