of = open("data/kiel_train/kielipankki.train")
nf = open("data/kiel_train/vocab.txt",'w')

outer_counter=0
inner_counter=0
lengths_ofline=[]
vocab=[]
for line in of:
    if len(line)<300:
        words=line.split()
        for word in words:
            if word not in vocab:
                vocab.append(word)
                nf.write(word+'\n')
                inner_counter+=1
    #lengths_ofline.append(len(line))
#percentage_over_ = [(i,np.sum(np.asarray(lengths_ofline)>i) / len(lengths_ofline)*100) for i in range(100,1000,100)]
print(inner_counter)
of.close()
nf.close()
