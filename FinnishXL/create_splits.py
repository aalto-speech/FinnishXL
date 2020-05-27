nf=open('yle_test_new','w')
with open('/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/yle_test_new_with_id', "r", encoding="utf-8") as reader:
   while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append(' ')
            Splitted[1]='<UNK>'
            print(Splitted[0],Splitted[1])
        nf.write(Splitted[1]+'\n')
nf.close()