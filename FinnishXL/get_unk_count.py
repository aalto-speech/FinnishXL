# coding: utf-8
import argparse
import time
import math
import os, sys

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_data/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='Ktrain',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8','Ktrain'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=150,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, default='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/',
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda")

# Get logger
# logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
#                      log_=not args.no_log)

# Load dataset
all_ids=[]
space_counter=0
nf= open("data/cp_kiel_train3/test_tr_50_nbest.txt",'w')
with open('/m/triton/scratch/elec/puhe/p/jaina5/tamas_lattice/text', "r", encoding="utf-8") as reader:
#with open('yle_nbest_50', "r", encoding="utf-8") as reader:
     while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('<S>')
        all_ids.append(Splitted[0])
        nf.write(Splitted[1]+'\n')
nf.close()
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

encoded_sent=corpus.vocab.encode_file(path='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/cp_kiel_train3/test_tr_50_nbest.txt',add_double_eos=True)
counter=0
sentences=[]
af=open("unk_tamas.txt",'w')
for idx,sent in enumerate(encoded_sent):

    if 30811 in sent:
        af.write(all_ids[idx]+' '+corpus.vocab.convert_to_sent(sent)+'\n')
af.close()
#print(counter)