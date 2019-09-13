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
        --d_head 60 \
        --d_inner 2048 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 10000 \
        --max_step 500000 \
        --lr 0.00025 \
        --tgt_len 200 \
        --mem_len 100 \
        --eval_tgt_len 32 \
        --batch_size 22 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data /m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_train2/ \
        --dataset Ktrain \
        --batch_size 64 \
        --tgt_len 200 \
        --mem_len 0 \
        --split test \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi
