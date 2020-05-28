#%%
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
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
work_dirs_72=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-135739/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-150831/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191022-134510/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191028-092926/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191105-144908/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191112-103726/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191119-133110/log.txt']
work_dirs_64=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-145634/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-150708/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191022-134625/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191028-092841/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191105-144926/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191112-102012/log.txt']
work_dirs_48=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-114941/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-150935/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191022-134353/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191028-093033/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191105-144751/log.txt']
work_dirs_32=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-114108/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-151152/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191022-134318/log.txt']
work_dirs_24=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-112931/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-151332/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191022-132640/log.txt']
work_dirs_16=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-112556/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-151403/log.txt']
work_dirs_8=['/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-112506/log.txt','/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191014-151746/log.txt']
work_dirs=[work_dirs_72,work_dirs_64,work_dirs_48,work_dirs_32,work_dirs_24,work_dirs_16,work_dirs_8]
ytick=np.arange(4,11,1)
fig, ax = plt.subplots()
color=iter(plt.cm.rainbow(np.linspace(0,1,8)))
labels=iter(['72','64','48','32','24','16','8'])
for sub_work_dir in work_dirs:
    epochs=[]
    learning_rates=[]
    steps=[]
    training_loss=[]
    training_ppl=[]
    msbatch=[]
    valid_step=[]
    valid_loss=[]
    valid_ppl=[]
    colo=next(color)
    label=next(labels)
    for work_dir in sub_work_dir:
        with open(work_dir, "r", encoding="utf-8") as reader:
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

    #plt.plot(np.array(steps),np.array(training_ppl),c=colo,label=label)
    plt.plot(np.array(steps[120:]),np.array(training_ppl[120:]),c=colo,label=label)
plt.title('Training Perplexity of T-XL models') 
plt.ylabel('Perplexity')
plt.xlabel('Steps')   
plt.legend()
plt.show()
#sns.lineplot(epochs,training_loss)

plt.savefig('train_ppl_midway_plots_txl.png',dpi=400)

