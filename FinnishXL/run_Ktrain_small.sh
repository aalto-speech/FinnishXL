#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_train/ \
        --dataset Ktrain \
        --n_layer 24 \
        --d_model 128 \
        --n_head 8 \
        --d_head 40 \
        --d_inner 512 \
        --dropout 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --warmup_step 30000 \
        --max_step 1200000 \
        --lr 0.00025 \
        --tgt_len 250 \
        --mem_len 0 \
        --eval_tgt_len 250 \
        --batch_size 22 \
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
