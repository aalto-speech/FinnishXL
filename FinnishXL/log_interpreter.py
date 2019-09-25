import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
epochs=[]
learning_rates=[]
steps=[]
training_loss=[]
training_ppl=[]
msbatch=[]
valid_step=[]
valid_loss=[]
valid_ppl=[]
counter=0
work_dirs=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20190913-122106/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20190828-114732/log.txt']
ytick=np.arange(4,11,1)
with open(work_dirs[0], "r", encoding="utf-8") as reader:
    for _ in range(67):
        next(reader)
    lines = reader.readlines()
    for line in lines:
        line = line.strip()
        if '--' in line:
            continue
        if '==' in line:
            continue
        if 'Eval' in line:
            split_line=line.split(' ')
            filter_split = list(filter(lambda a: a != '', split_line))
            filter_split = list(filter(lambda a: a != '|', filter_split))
            valid_step.append(filter_split[4])
            valid_loss.append(filter_split[9])
            valid_ppl.append(filter_split[11])
            continue
        if 'Exiting' in line:
            continue
        if 'End' in line:
            continue    
        split_line=line.split(' ')
        filter_split = list(filter(lambda a: a != '', split_line))
        filter_split = list(filter(lambda a: a != '|', filter_split))
        epochs.append(float(filter_split[1]))
        steps.append(float(filter_split[3]))
        learning_rates.append(float(filter_split[7]))
        training_loss.append(float(filter_split[11]))
        training_ppl.append(float(filter_split[13]))
fig, ax = plt.subplots()
plt.plot(np.array(steps),np.array(training_loss),'r-')
epochs=[]
learning_rates=[]
steps=[]
training_loss=[]
training_ppl=[]
msbatch=[]
valid_step=[]
valid_loss=[]
valid_ppl=[]
with open(work_dirs[1], "r", encoding="utf-8") as reader:
    for _ in range(67):
        next(reader)
    lines = reader.readlines()
    for line in lines:
        line = line.strip()
        if '--' in line:
            continue
        if '==' in line:
            continue
        if 'Eval' in line:
            split_line=line.split(' ')
            filter_split = list(filter(lambda a: a != '', split_line))
            filter_split = list(filter(lambda a: a != '|', filter_split))
            valid_step.append(filter_split[4])
            valid_loss.append(filter_split[9])
            valid_ppl.append(filter_split[11])
            continue
        if 'Exiting' in line:
            continue
        if 'End' in line:
            continue    
        split_line=line.split(' ')
        filter_split = list(filter(lambda a: a != '', split_line))
        filter_split = list(filter(lambda a: a != '|', filter_split))
        epochs.append(float(filter_split[1]))
        steps.append(float(filter_split[3]))
        learning_rates.append(float(filter_split[7]))
        training_loss.append(float(filter_split[11]))
        training_ppl.append(float(filter_split[13]))

plt.plot(np.array(steps),np.array(training_loss),'b-')
every_nth = 10
# for n, label in enumerate(ax.yaxis.get_ticklabels()):
#     if n % every_nth != 0:
#         label.set_visible(False)
# for n, label in enumerate(ax.xaxis.get_ticklabels()):
#     if n % 20 != 0:
#         label.set_visible(False)

locs, labels = plt.yticks()
#print(locs,labels)
plt.show()
#sns.lineplot(epochs,training_loss)