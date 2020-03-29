#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:45:56 2020

@author: george
"""

import os
import pandas as pd
import argparse
import numpy as np

George = False

if George:
	os.chdir("/home/george/Desktop/extreme-gnns/results")
else:
	os.chdir("/home/h/Documents/Japet/Extreme-GNNs/results")

perc = {}
for mod in ["gcn","gat","sage"]:#,"gat_deg","gat_core"]:
    for ds in  ['photo','computers',"Cora","CiteSeer","PubMed"]:
        x = pd.read_csv("res_val_"+mod+"_"+ds+".csv")
        # based on summation
        l = x["diff_r"]+x["diff_e"]        
        perc[mod+"_"+ds] = np.argmax(l.values)


    
for mod in ["gcn","gat","sage"]:#,"gat_deg","gat_core"]:
    for ds in  ['photo','computers',"Cora","CiteSeer","PubMed"]:
        x = pd.read_csv("res_test_"+mod+"_"+ds+".csv")
        x = x.iloc[perc[mod+"_"+ds],:]
        print(mod+"_"+ds)
        print(x["diff_r"])
        print(x["diff_e"])
        print(x[np.argmax(l.values)])


