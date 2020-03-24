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


os.chdir("/home/george/Desktop/extreme-gnns/log") 

for ext in ["test","val"]:
    for mod in ["gat","gcn","sage"]:
        for ds in  ['CiteSeer','Cora','PubMed']:
            df = pd.read_csv("logw_"+ ext+"_"+mod+"_"+ds+".csv")
            #"logw_"+ ext+"_"+mod+"_"+str(i)+"_"+ds+"
            df["perc"]= df["perc"]+2
            
            try:
                reg = df["test_r"]-df["test_br"]
                ex = df["test_e"]-df["test_be"]
            except:
                reg = df["val_r"]-df["val_br"]
                ex = df["val_e"]-df["val_be"]
            df["diff_r"] = reg
            df["diff_e"] = ex
            df.to_csv("../results/res_"+ ext+"_"+mod+"_"+ds+".csv",index=False)
        
      
    
    