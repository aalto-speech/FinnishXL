#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train_schedule_restart.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_data/ \
        --dataset Ktrain \
        --n_layer 18 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 4096 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --attn_type 2 \
        --optim adam \
        --warmup_step 20000 \
        --max_step 500000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 0 \
        --eval_tgt_len 32 \
        --batch_size 224 \
        --batch_chunk 4 \
        --restart \
        --restart_dir /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20200402-201111 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset Ktrain \
        --batch_size 512 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi
