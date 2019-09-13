all_ids=[]
space_counter=0

with open('yle_nbest_1000', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('<S>')
            space_counter+=1
        all_ids.append(Splitted[0])
print(space_counter)