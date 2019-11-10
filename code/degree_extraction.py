#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:52:43 2019

@author: george
"""

import os
import networkx as nx
import numpy as np
import scipy.io


os.chdir("/home/george/Desktop/mlgraphs/Data")

#---- edgelist files
dat_n = "cora.edges"
dat = nx.read_edgelist(dat_n,comments="#")
dat = nx.read_weighted_edgelist(dat_n,comments="#",delimiter=",")
dat_n = dat_n.replace(".edges",".txt")

#---- .mat files
dat_n = "Flickr.mat"
mat = scipy.io.loadmat(dat_n)
arr = np.array(mat)
d = mat["Network"]
dat_n = dat_n.replace(".mat",".txt")
dat = nx.Graph(d)

x = [j for i,j in dat.degree]
uni = np.unique(x)
f = open("degrees/deg_"+dat_n,"w")

for i in uni:
    f.write(str(i)+","+str(x.count(i))+"\n" ) 

f.close()                      



                       
