all_ids=[]
space_counter=0
nf= open("/m/triton/scratch/elec/puhe/p/jaina5/tamas_lattice1best_11LMWT/lm_cost",'w')
with open('/m/triton/scratch/elec/puhe/p/jaina5/tamas_lattice1best_11LMWT/text', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('<S>')
        all_ids.append(Splitted[0])
        nf.write(Splitted[0]+' 10'+'\n')
nf.close()