from functools import reduce
lengths=[]
with open('/m/triton/scratch/work/jaina5/kaldi/egs/yle_rescore/s5/hypoth_final_trxl_aff.txt', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('<S>')
            print(Splitted[0])
        lengths.append(len(Splitted[1].split()))
nf=open('yle-dev-new')
lengths_2=[]
for line in nf:
    line = line.strip()
    lengths_2.append(len(line.split()))
print(reduce(lambda x, y: float(x) + float(y),lengths ) / len(lengths))
print(reduce(lambda x, y: float(x) + float(y),lengths_2 ) / len(lengths_2))