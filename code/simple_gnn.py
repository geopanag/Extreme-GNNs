#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:56:35 2019

@author: george
"""

#import graph_nets as gn
#import sonnet as snt

import os
import pandas as pd

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import igraph as ig
import scipy.io
import networkx as nx
import numpy as np



class gnn(torch.nn.Module):
    #------- Define the neural architecture
    def __init__(self,num_features,num_classes,hidden_size):
        super(gnn, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes+1)

    def forward(self, dat):
        x, edge_index = dat.x, dat.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    

os.chdir("/home/george/Desktop/mlgraphs/Data")
           
 
#------- Parameters
hidden_size = 10

x = np.load("cora_features.npy")
num_features = x.shape[1]

labs =  pd.read_csv("cora_node_labels.txt",header=None)
labs = labs.set_index(0)
num_classes = len(labs[1].unique())

#------- Read the data
#d = pd.read_csv("cora_edges.txt",header=None)
#d.to_csv("cora_net.txt",sep="\t",index=False,header=False)
name = "cora_net.txt"
g = ig.Graph.Read_Ncol(name)
names = g.vs()["name"]
num_node = len(g.vs())

#------- Create the input to the algorithm
source = []
target = []
for i in g.es():
    source.append(i.tuple[0])
    target.append(i.tuple[1])
    
edge_index = torch.tensor([source,target])

#------- Define random 10% of nodes as train labels
train_idx = np.random.choice(num_node, round(10*num_node/100) ) 
train_mask = torch.tensor([True if i in train_idx else False for i in range(0,num_node)] )
test_mask =  torch.tensor([True if i not in train_idx else False for i in range(0,num_node)] )


#------- Define the input of the dataset
#y = torch.FloatTensor([np.random.choice(range(0,num_classes),sum(train_mask))])
y = torch.tensor(labs[1].values.tolist())
x = torch.tensor(x)

dat = Data(edge_index = edge_index, test_mask = test_mask, train_mask = train_mask, x = x , y = y)    


#------ Set labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = gnn(num_features,num_classes,hidden_size)
model = gnn.to(device)


#------ Run training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(dat)
    loss = F.nll_loss(out[dat.train_mask], dat.y[dat.train_mask])
    loss.backward()
    optimizer.step()
    
    
#------ Evaluate
model.eval()
_, pred = model(dat).max(dim=1)
correct = float (pred[dat.test_mask].eq(dat.y[dat.test_mask]).sum().item())
acc = correct / dat.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
    
    
    
    
    
#--------------------------------------- rough
    
#x = np.save("cora_features",x)
   
#model.eval()
#_, pred = model(dat).max(dim=1)
#correct = float (pred[dat.test_mask].eq(dat.y[dat.test_mask]).sum().item())
#acc = correct / dat.test_mask.sum().item()
#print('Accuracy: {:.4f}'.format(acc))


# Read datasets
#dat_n = "Flickr.mat"
#mat = scipy.io.loadmat(dat_n)
#arr = np.array(mat)
#d = mat["Network"]
#dat = nx.Graph(d)

#dataset = Planetoid(root='/tmp/Cora', name='Cora')
#data = dataset[0].to(device)

#dat_n = "cora_edges.txt"
#net = nx.read_weighted_edgelist(dat_n,comments="#",delimiter=",")
#labs = pd.read_csv("cora_node_labels.txt")             
#g = ig.Graph.Read_Ncol("Wiki-Vote.txt")


                                

