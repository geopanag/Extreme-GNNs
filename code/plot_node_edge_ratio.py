#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:02:34 2020

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
from torch_geometric.nn import GATConv
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import matplotlib.pyplot as plt   


from torch_geometric.datasets import Amazon, Coauthor

from sklearn.metrics import roc_auc_score
import igraph as ig
import scipy.io
import networkx as nx
import numpy as np

import torch_geometric.transforms as T

from collections import Counter
import operator




def separate_indices(samples,dic,thres):    
    # Separate the indices of the data that are extreme and regular
    idx_e = [name  for name in samples  if dic[name]>=thres ] 
    extremes  = np.repeat(False,len(names)) 
    extremes[idx_e] = True 
    
    idx_r = [name  for name in samples  if dic[name]<thres ] 
    regulars = np.repeat(False,len(names)) 
    regulars[idx_r] = True 
    
    #----- check the number of edges (e.g. the structure contained) in the extreme nodes compared to regular ones
    
    return len(idx_e),len(idx_r), sum([dic[i] for i in idx_e]), sum([dic[i] for i in idx_r]),extremes, regulars



def find_thres_naive(deg_val,percentile):
    uni = np.unique(deg_val)
    perc = 0
    for i in uni:
        perc+=deg_val.count(i)
        if(perc*100/len(deg_val)>=percentile):
            break
    return i


def get_label(train_extremes,y):
    d = dict(Counter([y[i] for i in np.where(train_extremes)[0]]))
    return  max(d.items(), key=operator.itemgetter(1))[0]

 
if __name__ == '__main__': 
    ds = "Photo"
    path = osp.join(osp.dirname(
                    osp.realpath("/home/george/Desktop/extreme-gnns/data")), 'data', ds) #
 
    dataset = Amazon(path,ds,T.NormalizeFeatures())
    #Planetoid(path, ds, T.NormalizeFeatures())
    data = dataset[0]
    edgelist = data.edge_index.numpy()
    
    g = nx.DiGraph()
    # because some nodes are not in the edgelists
    g.add_nodes_from(range(max(max(edgelist[0]),max(edgelist[1]))))
    
    for i in range(len(edgelist[0])):
        g.add_edge(edgelist[0][i],edgelist[1][i])
    
    # degreee based
    degs = dict(g.degree())
    names = list(g.nodes())
    
    test_set_size = len(names)/2
            
    
    percentile = 90
    deg_val = list(degs.values())
    
    to_plot = f = open("plot_ratios_"+ds+".csv","w")
    to_plot.write("percentile,e_nodes,r_nodes,e_edges,r_edges,c1,c2\n")
    
    class_sample = 20
    
    
    
    for i in range(10):
        thres = find_thres_naive(deg_val,percentile)
        #e_node=r_node=e_edges=r_edges=c1 =c2 = 0    
        for exp in range(10):
                    
            test_samples = np.random.choice([i for i in range(len(names))], round(test_set_size),replace=False)
            
            y = data.y.numpy()
            y = y[~test_samples]
            
            
            #---- 20 samples per class
            train_samples = []
            for cl in np.unique(y):
                train_samples.extend(np.random.choice(np.where(y==cl)[0],
                                                      class_sample,replace =False))
             
            #train_samples = [i for i in np.where(data.train_mask.numpy())[0] ] #np.random.choice( [i for i in range(len(names))] ,  round(20*len(names)/100),replace=False) 
            #idx = [names.index(i) for i in extreme_names]
          
            #e_n, r_n, e_e, r_e ,train_extremes,_ =  separate_indices(train_samples,degs,thres)
            e_node, r_node, e_edges, r_edges  ,train_extremes,_ =  separate_indices(train_samples,degs,thres)
            #e_node += e_n
            #r_node +=r_n
            #e_edges +=e_e
            #r_edges += r_e
            
            y = data.y.numpy()
            extreme_train_label = get_label(train_extremes,y)
            y = np.array([i==extreme_train_label for i in list(y)]).astype(int)
            
            c1 = list(y).count(0)
            c2 = list(y).count(1)
        
            to_plot.write(str(percentile)+","+str(int(e_node))+","+str(int(r_node))+","+str(int(e_edges))+","+str(int(r_edges))+","+str(int(c1))+","+str(int(c2))+"\n")
            
        percentile = percentile-2
        
    to_plot.close()
    
            





