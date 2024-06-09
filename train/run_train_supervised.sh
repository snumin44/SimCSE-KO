#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'klue/bert-base' \
        --tokenizer 'klue/bert-base' \
    	--train_data '../data/kor_nli_triplets.csv' \
        --valid_data 'kor_dev' \
    	--output_path '../output/supervised_simcse' \
    	--epochs 1 \
        --batch_size 256 \
        --max_length 64 \
        --dropout 0.1 \
        --pooler 'cls' \
        --eval_strategy 'steps' \
        --eval_step 100 \
        --amp \
        --padding  \
        --truncation \
        --shuffle 