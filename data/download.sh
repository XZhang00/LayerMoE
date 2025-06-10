export HF_HOME="your_path/.cache"
export XDG_CACHE_HOME="your_path/.cache"
export HF_HUB_ENABLE_HF_TRANSFER=1



# languages: el/hu/tr/bn/hi/ne/es  (only the first 30 parquet files following MoE-LPR)
huggingface-cli download \
    --token=yours \
    --repo-type dataset \
    --resume-download uonlp/CulturaX \
    --include "el*" \
    --local-dir your_path/Datasets/CulturaX-per30_parquet \
    --local-dir-use-symlinks False



# en
huggingface-cli download \
    --token=yours \
    --repo-type dataset \
    --resume-download cerebras/SlimPajama-627B \
    --include "train/chunk1*" \
    --local-dir your_path/Datasets/cerebras/SlimPajama-627B \
    --local-dir-use-symlinks False


# zh: SkyPile-150B/2023-14_zh_middle_0009
