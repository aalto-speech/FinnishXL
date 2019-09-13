lengths=[]
with open('yle_nbest_1000', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('<S>')
        lengths.append(len(Splitted[1].split()))
re=open("rescore_2.txt")
re2=open("rescore_hue_2.txt","w")
counter=0
for rescore in re:
    rescore=rescore.strip()
    rescore=rescore.split(" ",1)
    tmp_length=lengths[counter]
    divisor=(5+tmp_length)**0.5/(6)**0.5
    tmp_score=float(rescore[1])/divisor
    counter+=1
    re2.write(rescore[0]+" "+str(tmp_score)+"\n")
re.close()
re2.close()


