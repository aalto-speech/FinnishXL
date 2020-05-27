# For splitting the files into pieces
gawk '
    BEGIN {srand()}
    {f = FILENAME (rand() <= 0.8 ? ".80" : ".20"); print > f}
' file
# Some slurm commands
sbatch script.slrm
slurm ss|grep gpu
slurm p|grep gpu
sacct -j $JOB_ID -o comment%-100

#Kaldi commands
python kaldi-utensils/cutlery/rescore_nbest.py --lm-weight 11 /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_nbest_10 /m/triton/scratch/elec/puhe/p/jaina5/ac_cost.10best.aff /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/rescore_30layer_10.txt > 30layer_10nbest_hyp.txt

bash /scratch/work/rouhea1/kaldi-vanilla/egs/yle-dev-rescore/s5/get_yle_dev_wer.sh 30layer_10nbest_hyp.txt

bash /scratch/work/rouhea1/kaldi-vanilla/egs/yle-dev-rescore/s5/get_yle_dev_wer.sh --use_bootci true --compare rescore_50_20191119-133110_12.5  rescore_50_20200123-133425
find . -size +1G | cat >> .gitignore; awk '!NF || !seen[$0]++' .gitignore


 for f in x[0-9]*; do mv "$f" "$((10#${f#x}+1))"; done

 split -l 10000 -d -a 3 yle_nbest_1000_sentences
 
 for i in *; do mv "$i" x"$i"; done

 :%s/[{}¬×!?.#μ‰€舗舡]//g

 python kaldi-utensils/cutlery/rescore_nbest_interpolate.py --lm-weight 11 --lm-weight2 -1.5 /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_nbest_1000 /m/triton/scratch/elec/puhe/p/jaina5/ac_cost.1000best.aff /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/rescore_72layer_1000nbest_yle_20191119-133110.txt lm_cost.psmit_rescore_1000  > interpolatepsmit_11_1.5_nbest1000.hyp

compute-wer-bootci --mode=present ark:/scratch/work/psmit/chars-fin-2017/am/data/yle-dev-new/text ark:LSTM_50_nbest_after_nbest1000.hyp ark:rescore_interpolate/interpolate_psmit_10_3.txt 

FRILAND_20120623123200_00W2A20120623123148_01-011-36 <UNK>
FRILAND_20120625112917_00X0L20120625112905_02-015-8 <UNK>
FRILAND_20120625112917_00X0L20120625112905_04-019-306 <UNK>
FRILAND_20120626164543_000ML20120626164531_02-013-73 <UNK>
yle2012-1011-1803-048-12 <UNK>
yle2012-1211-0905-016-55 <UNK>
yle2012-1511-1903-009-81 <UNK>
yle2012-2410-1903-016-2 <UNK>

/scratch/work/psmit/chars-fin-2017/am/data/yle-test-new/text

python kaldi-utensils/cutlery/rescore_nbest_interpolate.py --lm-weight 12 --lm-weight2 -12 /scratch/elec/puhe/p/jaina5/
transformer-xl/FinnishXL/yle_nbest_1000 /m/triton/scratch/elec/puhe/p/jaina5/ac_cost.1000best.aff /scratch/elec/puhe/p/jaina5/
transformer-xl/FinnishXL/rescore_72layer_1000nbest_yle_20191119-133110.txt lm_cost.psmit_rescore_1000  > interpolatepsmit_12_1
2_nbest1000.hyp

python kaldi-utensils/cutlery/rescore_nbest.py --lm-weight 9 /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_test_nbest_50 /scratch/elec/puhe/p/jaina5/yle_test_ac_cost_50 /scratch/work/jaina5/Bert/FinnishBert/outputckpoints_16042020_10_32batch/Segment50_yle_test_without_alpha/lm_cost_yle_test_without_alpha > yle_test_BERT_withoutalpha_10layers_50.hyp