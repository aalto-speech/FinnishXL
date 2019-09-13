#!/bin/bash
morfessor-segment -l /scratch/work/psmit/chars-fin-2017/lm/all/morfessor_f2_a0.001_tokens/model.bin \
  --encoding='utf-8' \
  --output-format="{analysis} " \
  --output-format-separator="+ +" \
  --output-newlines \
  /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/monolingual_wmt/news.fi.shuffled > \
  /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/data/monolingual_wmt/subword.news.fi.traincd 

 steps/lmrescore_theanolm_nbest.sh --cmd "slurm.pl --time 1:00:00 --mem 25G --gpu 1" --N 50 /scratch/work/psmit/chars-fin-2017/am/data/recog_langs/morfessor_f2_a0.001_tokens_aff_small/ /scratch/work/psmit/chars-fin-2017/am/data/lm/morfessor_f2_a0.001_tokens_aff/rescore/word+proj500+lstm1500+htanh1500x4+dropout0.2+softmax_e13.5_t468.h5 /scratch/work/psmit/chars-fin-2017/am/exp/chain/model/all_tdnn_blstm_9_a/decode1150_yle-dev-new_morfessor_f2_a0.001_tokens_aff_small rescore_psmit/