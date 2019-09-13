#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/cp_kiel_train/ \
        --dataset Ktrain \
        --n_layer 20 \
        --d_model 512 \
        --n_head 8 \
        --d_head 40 \
        --d_inner 2048 \
        --dropout 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --warmup_step 60000 \
        --max_step 400000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 512 \
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
