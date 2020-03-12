#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:40:40 2020

@author: george
"""



import os
import pandas as pd
import os.path as osp
import argparse

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import matplotlib.pyplot as plt   
import numpy as np


os.chdir("/home/dascim/panago/results") 

repetitions =10
for ds in  ['CiteSeer','Cora','PubMed']:
    df = pd.read_csv("logw_0_"+ds+".csv")
    for i in range(1,repetitions):
        df_t = pd.read_csv("logw_"+str(i) +"_"+ds+".csv")
        df = df + df_t
    df = df/10
    df["perc"]= df["perc"]+2
    
    reg = df["test_r"]-df["test_br"]
    ex = df["test_e"]-df["test_be"]
    s = ex+reg
    m = np.max(s)
    pos = s.values.argmax()
    perc = df.loc[pos,"perc"]
    print(ds+" improvement of :"+str(m)+" at percentile "+str(perc))
    
    #l = list(set(np.where(reg>0)[0]).intersection(set(np.where(ex>0)[0])))
    #reg= list(reg)
    #ex = list(ex)
    #for i in l:
    
    
    