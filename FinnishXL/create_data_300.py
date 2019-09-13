import numpy as np
#of = open("data/kiel_train/xaa")
#nf = open("data/kiel_train/train.txt",'w')
#of = open("data/kiel_train/xab")
#nf = open("data/kiel_train/valid.txt",'w')
of = open("data/kiel_train/test")
nf = open("data/kiel_train/test.txt",'w')

outer_counter=0
inner_counter=0
lengths_ofline=[]
for line in of:
    if len(line)<300:
        nf.write(line)
        inner_counter+=1
    #lengths_ofline.append(len(line))
#percentage_over_ = [(i,np.sum(np.asarray(lengths_ofline)>i) / len(lengths_ofline)*100) for i in range(100,1000,100)]
print(inner_counter)
of.close()
nf.close()

