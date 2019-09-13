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
parser.add_argument('--data', type=str, default='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/cp_kiel_train3/',
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
with open('yle_nbest_50', "r", encoding="utf-8") as reader:
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
# va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
#     device=device, ext_len=args.ext_len)
# te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
#     device=device, ext_len=args.ext_len)
# model_dirs=['20190810-202326','20190809-180349','20190729-225418','20190804-082026','20190812-114939','20190812-114939','20190812-125743',
# '20190806-212523','20190806-110744','20190812-115633','20190802-171559','20190730-144052','20190815-205709']
model_dirs=['20190816-112952','20190816-130551','20190818-175221','20190819-120901','20190820-122300','20190821-142959']
test_dirs=['20190726-150431','20190726-172009']
for model_dir in model_dirs:
    # Load the best saved model.
    with open(os.path.join(args.work_dir+model_dir+'/', 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model.backward_compatible()
    model = model.to(device)

    # logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
    #        args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

    #model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True

    ###############################################################################
    # Evaluation code
    ###############################################################################
    # def per_sentence():
    #     tokens=[]
    #     with open('/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/cp_kiel_train3/test.txt', "r", encoding="utf-8") as reader:
    #         while True:
    #             line = reader.readline()
    #             if not line:
    #                 break
    #             line = line.strip()
    #             tokens.append([line])
    #     return tokens
    re=open("rescore2/rescore_50_"+model_dir,'w')
    def rescore():
        encoded_sent=corpus.vocab.encode_file(path='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/cp_kiel_train3/test_tr_50_nbest.txt',add_double_eos=True)
        for idx,sent in enumerate(encoded_sent):
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
                loss=loss.sum()
                sent_rescore=''
                sent_rescore=all_ids[idx]+' '+str(loss.item())
                re.write(sent_rescore+'\n')
            
    rescore()
    re.close()




# # Run on test data.
# if args.split == 'all':
#     test_loss = evaluate(te_iter)
#     valid_loss = evaluate(va_iter)
# elif args.split == 'valid':
#     valid_loss = evaluate(va_iter)
#     test_loss = None
# elif args.split == 'test':
#     test_loss = evaluate(te_iter)
#     valid_loss = None

# def format_log(loss, split):
#     if args.dataset in ['enwik8', 'text8']:
#         log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
#             split, loss, loss / math.log(2))
#     else:
#         log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
#             split, loss, math.exp(loss))
#     return log_str

# log_str = ''
# if valid_loss is not None:
#     log_str += format_log(valid_loss, 'valid')
# if test_loss is not None:
#     log_str += format_log(test_loss, 'test')

# logging('=' * 100)
# logging(log_str)
# logging('=' * 100)
