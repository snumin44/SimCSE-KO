#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'klue/roberta-base' \
        --tokenizer 'klue/roberta-base' \
    	--train_data '../data/korean_wiki_1m.txt' \
        --valid_data 'kor_dev' \
    	--output_path '../output/unsupervised_simcse' \
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