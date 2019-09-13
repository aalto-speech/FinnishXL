#!/bin/bash

grep -o -E '\+*<*\w+\+*>*' data/kiel_train/kielipankki.train | sort -u -f > data/kiel_train/vocab_train.txt