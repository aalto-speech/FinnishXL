# coding: utf-8
import argparse
import time
import math
import os, sys

import torch

from data_utils import get_lm_corpus,LMShuffledIterator,LMOrderedIterator
from mem_transformer import MemTransformerLM

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8','Ktrain'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=200,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str,default='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/',
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda")

#Load dataset
##############################################################################
#Evaluation code
##############################################################################
corpus=get_lm_corpus('/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_data/','Ktrain')
#model_dirs=['20191119-133110','20191014-151746','20191014-151403','20191022-134318','20191105-144751','20191112-102012','20191119-133110','20190818-175221','20200407-201955']
model_dirs=['20200407-201955']
for model_dir in model_dirs:
    # Load the best saved model.
    with open(os.path.join(args.work_dir+model_dir+'/', 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model.backward_compatible()
    model = model.to(device)
    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True
    
    def rescore():
        #encoded_sent=corpus.vocab.encode_file(path='/m/triton/scratch/elec/puhe/p/jaina5/yle_test_new_text.segmented.padded',add_double_eos=True)
        encoded_sent=corpus.vocab.encode_file(path='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle-dev-new',add_double_eos=True)
        total_loss=0
        total_len=0
        for _,sent in enumerate(encoded_sent):
            streams = [None] * 1
            bptt=len(list(sent))-1
            data = torch.LongTensor(bptt, 1)
            target = torch.LongTensor(bptt, 1)
            model.reset_length(bptt, args.ext_len, args.mem_len)
            n_retain = 0        
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)
            for i in range(1):
                n_filled = 0
                
                while n_filled < bptt:
                    if streams[i] is None or len(streams[i]) <= 1:
                        streams[i] = sent
                    # number of new tokens to fill in
                    n_new = min(len(streams[i]) - 1, bptt - n_filled)
                    # first n_retain tokens are retained from last batch
                    data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                        streams[i][:n_new]
                    target[n_filled:n_filled+n_new, i] = \
                        streams[i][1:n_new+1]
                    streams[i] = streams[i][n_new:]
                    n_filled += n_new

            data = data.to(device)
            target = target.to(device)
            model.eval()
            mems=tuple()
            with torch.no_grad():
                ret = model(data, target,*mems)
                loss=ret[0]
                loss=loss.mean()
                total_loss+=bptt*loss.item()
                total_len+=bptt
        print("Total ppl for {} is {}".format(model_dir, math.exp(total_loss/total_len)))

            
    rescore()

