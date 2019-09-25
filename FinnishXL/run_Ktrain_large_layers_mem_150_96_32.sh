#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train_schedule_restart_96.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_data/ \
        --dataset Ktrain \
        --n_layer 96 \
        --d_model 256 \
        --n_head 8 \
        --d_head 40 \
        --d_inner 1024 \
        --dropout 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --warmup_step 0 \
        --max_step 200000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 512 \
        --batch_chunk 4 \
        --restart \
        --restart_dir /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20190913-122106 \
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
