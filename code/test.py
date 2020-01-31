#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:11:44 2020

@author: george
"""

import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import igraph as ig
import os
import pandas as pd
import numpy as np

os.chdir("/home/george/Desktop/extreme-gnns/data")
           
#------- Dataset & Parameters
hidden_size = 20
thres = 6.8399999999999981
lr = 0.01
n_epochs = 200


thresholds = eval(open('degree_thresholds.txt' , 'r').read())
        

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath("/home/george/Desktop/extreme-gnns")), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


fn = "cora-net"#"brazil-airports","europe-airports","usa-airports" ,
name = fn+".edgelist"
g = ig.Graph.Read_Ncol(name)
names = g.vs()["name"]
num_node = len(g.vs())

# degreee based
degs = g.degree()
dic = {i:j for i,j in zip(names,degs)}

labs =  pd.read_csv("labels-"+fn+".txt",sep=" ")
labs = labs.set_index("node")
num_classes = len(labs["label"].unique())

x = np.load("cora_features.npy")
num_features = x.shape[1]
thres = thresholds["cora-net"]
    
train_samples = np.random.choice( [i for i in range(len(names))] ,  round(20*len(names)/100),replace=False) 

source = []
target = []
for i in g.es():
    source.append(i.tuple[0])
    target.append(i.tuple[1])
    
edge_index = torch.tensor([source,target])

#------- Input to the extreme method
train_mask_e = torch.tensor(train_extremes)
test_mask_e =  torch.tensor(test_extremes)


#--- Input to the regular method 
train_mask_r = torch.tensor(train_regulars)
test_mask_r =  torch.tensor(test_regulars)#[not i for i in test_extremes])

#------- Define the input variables 
y = torch.tensor(labs[1].values.tolist())
x = torch.tensor(x)    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------------------------ Regular
dat = Data(edge_index = edge_index, test_mask = test_mask_r, 
            train_mask = train_mask_r, x = x , y = y)    

rgnn = gcn(num_features,num_classes,hidden_size)





