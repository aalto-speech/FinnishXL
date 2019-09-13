import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

corpus = get_lm_corpus('data/kiel_train/','Ktrain')
ntokens = len(corpus.vocab)
n_token = ntokens
device = torch.device('cuda')
eval_batch_size = 10
tr_iter = corpus.get_iterator('test', 64, 150,
    device=device, ext_len=0)