set -eu
for file in /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/rescore4/*; do
    base=$(basename $file); echo $base
    for weight in $(seq 4 15); do
        result_file=rescore/hypoth_final_trxl_aff_"$base"_"$weight".txt
        python kaldi-utensils/cutlery/rescore_nbest.py --lm-weight "$weight" \
        /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_nbest_50 /m/triton/scratch/elec/puhe/p/jaina5/ac_cost.50best.aff \
        "$file" > "$result_file"
        bash /scratch/work/rouhea1/kaldi-vanilla/egs/yle-dev-rescore/s5/get_yle_dev_wer.sh "$result_file" > results_rescore_4/"$base"_"$weight"
    done
done