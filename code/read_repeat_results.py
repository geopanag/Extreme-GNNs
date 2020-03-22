#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:53:21 2020

@author: george
"""
import os
import pandas as pd
import os.path as osp
import argparse
import numpy as np


os.chdir("/home/george/Desktop/extreme-gnns/log") 


repetitions = 10
for ext in ["test","val"]:
    for mod in ["gcn","gat","sage"]:#,"gat_deg","gat_core"]:
        for ds in  ['photo','computers']:
            print("logw_"+ ext+"_"+mod+"_0_"+ds+".csv")
            df = pd.read_csv("logw_"+ ext+"_"+mod+"_0_"+ds+".csv")
            for i in range(1,repetitions):
                try:
                    df_t = pd.read_csv("logw_"+mod+"_"+str(i) +"_"+ds+".csv")
                    df = df + df_t
                except:
                    continue
            df = df/(i+1)
            df["perc"]= df["perc"]+4
            
            try:
                reg = df["test_r"]-df["test_br"]
                ex = df["test_e"]-df["test_be"]
            except:
                reg = df["val_r"]-df["val_br"]
                ex = df["val_e"]-df["val_be"]
            df["diff_r"] = reg
            df["diff_e"] = ex
            #df["sum_diff"] = ex+reg
            df.to_csv("../results_amazon/res_"+ ext+"_"+mod+"_"+ds+".csv",index=False)
