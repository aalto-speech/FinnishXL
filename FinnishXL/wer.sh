set -eu
for file in /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/rescore_yle_test/rescore_yle_test_nbest50_*; do
    base=$(basename $file); echo $base
    for weight in $(seq 4 15); do
        result_file=rescore_yle_test/hypoth_final_trxl_aff_"$base"_"$weight".txt
        python kaldi-utensils/cutlery/rescore_nbest.py --lm-weight "$weight" \
        /scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_test_nbest_50 /m/triton/scratch/elec/puhe/p/jaina5/yle_test_ac_cost_50 \
        "$file" > "$result_file"
        bash /scratch/work/rouhea1/kaldi-vanilla/egs/yle-dev-rescore/s5/get_yle_test_wer.sh "$result_file" > results_rescore_yle_test/"$base"_"$weight"
    done
done