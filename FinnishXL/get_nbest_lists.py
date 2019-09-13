#nf=open('/m/triton/scratch/elec/puhe/p/jaina5/lm_cost.50best_trxl.aff','w')
#nf=open('/m/triton/scratch/elec/puhe/p/jaina5/ac_cost.50best.aff','w')
nf=open('yle_nbest_50','w')

nbest=50
#with open('rescore_2.txt', "r", encoding="utf-8") as reader:

#with open('/m/triton/scratch/elec/puhe/p/jaina5/ac_cost.100best.aff', "r", encoding="utf-8") as reader:
with open('yle_nbest_100', "r", encoding="utf-8") as reader:    
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append(' ')
        tempsplit=Splitted[:]
        if int(tempsplit[0].rsplit('-',1)[1]) <= nbest:
            nf.write(Splitted[0]+' '+Splitted[1]+'\n')
nf.close()