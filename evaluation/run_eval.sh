#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 eval.py \
    	--model '../output/unsupervised_simcse' \
        --tokenizer '../output/unsupervised_simcse' \
        --test_data 'klue_dev' \
    	--output_path '../output/result.txt' \
        --batch_size 256 \
        --max_length 64 \
        --pooler 'cls' \
