nf= open("psmit_lm_scores_ids_100",'w')
all_ids=[]
with open('/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_nbest_100', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('<S>')
        all_ids.append(Splitted[0])

scores=[]
with open('/scratch/work/jaina5/kaldi/egs/yle_rescore/s5/split_100_best_results/psmit_lm_scores_100', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line=line.strip()
        scores.append(line)
for idx,id in enumerate(all_ids):
    nf.write(id+' '+scores[idx]+'\n')
nf.close()

# nf= open("data/cp_kiel_train3/test_100best.txt",'w')
# with open('yle_nbest_100', "r", encoding="utf-8") as reader:
#     while True:
#         line = reader.readline()
#         if not line:
#             break
#         line = line.strip()
#         Splitted=line.split(" ", 1)
#         if len(Splitted) == 1:
#             Splitted.append('<S>')
#         nf.write(Splitted[1]+'\n')
# nf.close()