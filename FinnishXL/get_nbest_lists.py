#nf=open('/m/triton/scratch/elec/puhe/p/jaina5/lm_cost.50best_trxl.aff','w')
#nf=open('/m/triton/scratch/elec/puhe/p/jaina5/ac_cost.50best.aff','w')
nf=open('lstm_ac_cost_100','w')

nbest=100
#with open('rescore_2.txt', "r", encoding="utf-8") as reader:

with open('/m/triton/scratch/elec/puhe/p/jaina5/decode1150_yle-dev-new_morfessor_f2_a0.001_tokens_aff_rnn_interp_word+proj500+lstm1500+htanh1500x4+dropout0.2+softmax_e10.5_t365_i0.3_1000best/ac_cost', "r", encoding="utf-8") as reader:
#with open('/m/triton/scratch/elec/puhe/p/jaina5/decode1150_yle-dev-new_morfessor_f2_a0.001_tokens_aff_rnn_interp_word+proj500+lstm1500+htanh1500x4+dropout0.2+softmax_e10.5_t365_i0.3_1000best/text', "r", encoding="utf-8") as reader:    
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