#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_train/ \
        --dataset Ktrain \
        --n_layer 6 \
        --d_model 512 \
        --n_head 6 \
        --d_head 80 \
        --d_inner 2048 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 60000 \
        --max_step 1200000 \
        --lr 0.00025 \
        --tgt_len 250 \
        --mem_len 0 \
        --eval_tgt_len 250 \
        --restart \
        --restart_dir /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20190809-180349/ \
        --batch_size 32 \
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
