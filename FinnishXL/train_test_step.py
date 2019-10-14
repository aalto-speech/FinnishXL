import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

with open('/m/triton/scratch/elec/puhe/p/jaina5/transformer-xl/FinnishXL/-Ktrain/20191007-145634/trainstep.pt', 'rb') as f:
            train_step = torch.load(f)
            print(train_step)