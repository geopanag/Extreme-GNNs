
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


os.chdir("/home/george/Desktop/Extreme-GNNs/Data")


#---- edgelist files
#dat_n ="cora_net.txt"
#dat_n ="CA-HepPh.txt"
#dat_n ="CA-AstroPh.txt"
dat_n ="Wiki-Vote.txt"

dat = nx.read_edgelist(dat_n,comments="#",delimiter="\t")
#dat_n = dat_n.replace(".edges",".txt")


#---- .mat files
#BlogCatalog.mat
#dat_n = "Flickr.mat"
dat_n = "BlogCatalog.mat"
mat = scipy.io.loadmat(dat_n)
arr = np.array(mat)
d = mat["Network"]
dat_n = dat_n.replace(".mat",".txt")
dat = nx.Graph(d)


dat.remove_edges_from(nx.selfloop_edges(dat))
x = [j for i,j in nx.core_number(dat).items()]
uni = np.unique(x)
f = open("cores/cores_"+dat_n,"w")

for i in uni:
    f.write(str(i)+","+str(x.count(i))+"\n" ) 

f.close()                      



                       
