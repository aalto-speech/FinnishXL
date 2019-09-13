#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_kiel_train.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/cp_kiel_train/ \
        --dataset Ktrain \
        --n_layer 10 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 80 \
        --d_inner 2048 \
        --dropout 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --warmup_step 50000 \
        --max_step 600000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 512 \
        --restart \
        --restart_dir /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20190729-161101/ \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_train2/ \
        --dataset Ktrain \
        --batch_size 64 \
        --tgt_len 150 \
        --mem_len 0 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi
