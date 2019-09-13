#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_train/ \
        --dataset Ktrain \
        --n_layer 48 \
        --d_model 512 \
        --n_head 8 \
        --d_head 80 \
        --d_inner 2048 \
        --dropout 0.00 \
        --dropatt 0.05 \
        --optim adam \
        --attn_type 2 \
        --warmup_step 80000 \
        --max_step 1200000 \
        --lr 0.00025 \
        --tgt_len 150 \
        --mem_len 0 \
        --eval_tgt_len 150 \
        --batch_size 64 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --batch_size 8 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi
