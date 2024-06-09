#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 eval.py \
    	--model '../output/supervised_simcse' \
        --tokenizer '../output/supervised_simcse' \
        --test_data 'kor_test' \
    	--output_path '../output/result.txt' \
        --batch_size 256 \
        --max_length 64 \
        --pooler 'cls' \
        --padding  \
        --truncation \
        --shuffle 