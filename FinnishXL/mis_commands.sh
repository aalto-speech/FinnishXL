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