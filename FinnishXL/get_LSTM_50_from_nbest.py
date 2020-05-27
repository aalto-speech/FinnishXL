#!/usr/bin/env python3
# This rescores nbest lists
from operator import itemgetter

def parse_uttid_and_n(nbest_id):
  # returns uttid, n
  return nbest_id.rsplit("-", maxsplit=1)

def read_cost_file(path):
  costs = {} #{<uttid>:{<n>:cost}}
  with open(path) as fi:
    for line in fi:
      nbest_id, cost = line.strip().split()
      uttid, n = parse_uttid_and_n(nbest_id)
      costs.setdefault(uttid, {})[n] = float(cost)
  return costs

def read_nbest_text(path):
  hyps = {} #{<uttid>:{<n>:text}}
  with open(path) as fi:
    for line in fi:
      try:
        nbest_id, text = line.strip().split(maxsplit=1)
      except ValueError: #empty text
        nbest_id = line.strip()
        text = ""
      uttid, n = parse_uttid_and_n(nbest_id)
      hyps.setdefault(uttid, {})[n] = text
  return hyps

def find_lowest_costs(costs_list, weights):
  #costs should be a list of dictionaries, weights should be a list of weights for each dictionary
  lowest_costs = [] 
  lowest_cost_sorted= []#[(uttid,n,weightedcost)]
  for uttid in costs_list[0]:
    try: 
      for n in costs_list[0][uttid]:
        weighted_cost = sum(weights[i] * costs[uttid][n] for i, costs in enumerate(costs_list))
        #if uttid not in lowest_costs or lowest_costs[uttid][1] > weighted_cost:
        lowest_costs.append((uttid, n, weighted_cost))
    except KeyError:
      raise(KeyError("NBest-id "+uttid+"-"+n+" is not found in all cost lists!"))
    lowest_costs.sort(key=itemgetter(2))
    if len(lowest_costs) > 50:
        lowest_cost_sorted.append(lowest_costs[:50])
    else :
        lowest_cost_sorted.append(lowest_costs)
    lowest_costs=[]

  return lowest_cost_sorted

def choose_hypotheses(hyps, lowest_costs,ac_costs):
    nf=open('/m/triton/scratch/elec/puhe/p/jaina5/Psmit_lstm_yle_test_50_nbest_hyp','w')
    af=open('/m/triton/scratch/elec/puhe/p/jaina5/Psmit_lstm_yle_test_50_nbest_ac_costs','w')
    for i,best_tuples in enumerate(lowest_costs):
        #chosen[uttid] = hyps[uttid][n]
        for j,best_tuple in enumerate(best_tuples):
            text=hyps[best_tuple[0]][best_tuple[1]]
            ac_cost=ac_costs[best_tuple[0]][best_tuple[1]]
            nf.write(best_tuple[0]+"-"+best_tuple[1]+" "+text+"\n")
            af.write(best_tuple[0]+"-"+best_tuple[1]+" "+str(ac_cost)+"\n")
    nf.close()
    af.close()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser("Takes in kaldi-style nbest lists and associated costs files, outputs the chosen hypotheses")
  parser.add_argument("text", help = "hypotheses file, format: <uttid>-<n> <token1> <token2>...")
  parser.add_argument("ac_cost", help = "acoustic costs file, format: <uttid>-<n> <cost>")
  parser.add_argument("lm_cost", help = "language model costs file, format: <uttid>-<n> <cost>")
  parser.add_argument("--lm-weight", default = 13.0, type=float, help = "the language model weight, which multiplies the lm costs")
  args = parser.parse_args()
  hyps = read_nbest_text(args.text)
  ac_costs = read_cost_file(args.ac_cost)
  lm_costs = read_cost_file(args.lm_cost)
  lowest_costs = find_lowest_costs([ac_costs, lm_costs], [1.0, args.lm_weight])
  choose_hypotheses(hyps, lowest_costs,ac_costs)

