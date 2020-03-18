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
import numpy as np

os.chdir("/home/george/Desktop/extreme-gnns/results_new") 


for mod in ["gat","gcn"]:
    for ds in  ['CiteSeer','Cora','PubMed']:
        df = pd.read_csv("logw_"+mod+"_"+ds+".csv")
        df["perc"]= df["perc"]+4
        
        reg = df["test_r"]-df["test_br"]
        ex = df["test_e"]-df["test_be"]
        df["diff_r"] = reg
        df["diff_e"] = ex
        df.to_csv("../results_new/res_"+mod+"_"+ds+".csv",index=False)
    
      
    
    