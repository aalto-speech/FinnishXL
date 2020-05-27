# coding: utf-8
import argparse
import time
import math
import os, sys

import torch

from data_utils import get_lm_corpus,LMShuffledIterator,LMOrderedIterator
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

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
parser.add_argument('--mem_len', type=int, default=100,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str,default='/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20200407-201955/',
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

#Load dataset
##############################################################################
#Evaluation code
##############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
    return total_loss/total_len
corpus=get_lm_corpus('/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/kiel_data/','Ktrain')
ntokens = len(corpus.vocab)
#Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)
#model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

te_iter = corpus.get_iterator('test', 128, 200,device=device, ext_len=args.ext_len,shuffle=False)
te_loss=evaluate(te_iter)
print(te_loss)
# def rescore(batch_sentenes):
#     total_loss=0
#     total_len=0
#     for idx,sent in enumerate(batch_sentenes):
#         temp_sent=corpus.vocab.encode_text_batch(sent)
#         tmp_iter=LMShuffledIterator(temp_sent,1,len(temp_sent[0])-1,device=device,shuffle=False)
#         loss,length=evaluate(tmp_iter)
#         sub_word_loss=loss/length
#         #print(length,loss,sub_word_loss)
#         total_loss+=loss
#         total_len+=length
#         if total_len > 1000:
#             break
#         #nf.write(str(tmp_id)+' '+str(loss)+'\n')
#     print(total_len,total_loss,total_loss/total_len)
# tokens=[]
# with open('data/kiel_train2/test.txt', "r", encoding="utf-8") as reader:
#     while True:
#         line = reader.readline()
#         if not line:
#             break
#         line = line.strip()
#         tokens.append([line])
# rescore(tokens)
# vocab=[]
# with open('data/kiel_train2/vocab_train.txt', "r", encoding="utf-8") as reader:
#     while True:
#         line = reader.readline()
#         if not line:
#             break
#         line = line.strip()
#         vocab.append(line)
# counter=0
# for line in tokens:
#     words=line.split(' ')
#     for word in words:
#     #print(word)
#         if word not in vocab:
#             counter+=1
# print(counter)

# va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
#     device=device, ext_len=args.ext_len)
# te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
#     device=device, ext_len=args.ext_len)
# batch_sentenes=[['ulkomaan lenno+ +t sitä vastoin pyritään hoitamaan'],['<S>']]
# for idx,sent in enumerate(batch_sentenes):
#     print(sent)
#     temp_sent=corpus.vocab.encode_text_batch(sent)
#     print(temp_sent)
#     print(len(temp_sent[0]))
#     tmp_iter=LMShuffledIterator(temp_sent,1,len(temp_sent[0])-1,device=device)
#     loss=evaluate(tmp_iter)
#     print(loss)


        
        
# logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
#        args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

# model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
# if args.clamp_len > 0:
#     model.clamp_len = args.clamp_len
# if args.same_length:
#     model.same_length = True

# ###############################################################################
# # Evaluation code
# ###############################################################################
# def evaluate(eval_iter):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     total_len, total_loss = 0, 0.
#     start_time = time.time()
#     with torch.no_grad():
#         mems = tuple()
#         for idx, (data, target, seq_len) in enumerate(eval_iter):
#             ret = model(data, target, *mems)
#             loss, mems = ret[0], ret[1:]
#             loss = loss.mean()
#             total_loss += seq_len * loss.item()
#             total_len += seq_len
#         total_time = time.time() - start_time
#     #logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
#             #total_time, 1000 * total_time / (idx+1)))
#     return total_loss / total_len

# # Run on test data.
# # if args.split == 'all':
# #     test_loss = evaluate(te_iter)
# #     valid_loss = evaluate(va_iter)
# # elif args.split == 'valid':
# #     valid_loss = evaluate(va_iter)
# #     test_loss = None
# # elif args.split == 'test':
# # test_loss = evaluate(te_iter)
# # valid_loss = None

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
