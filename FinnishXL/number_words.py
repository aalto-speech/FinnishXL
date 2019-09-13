of = open("data/kiel_train/test.txt")
word_counter=0
for line in of:
    line=line.strip()
    words=line.split()
    for word in words:
        #if(word[len(word)-1] not in '+'):
            word_counter+=1
print(word_counter)