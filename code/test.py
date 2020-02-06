#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:11:44 2020

@author: george
"""

import os.path as osp
import argparse
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import igraph as ig
import os
import pandas as pd
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


    
os.chdir("/home/george/Desktop/extreme-gnns/data")

# standard way
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath("/home/george/Desktop/extreme-gnns")), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
   
data 
train = list(data.train_mask.numpy())
val = list(data.val_mask.numpy())
test = list(data.test_mask.numpy())
np.where(train)
np.where(val)
np.where(test)

t= data.x.numpy()
(t!=x).any()

#------- Dataset & Parameters
hidden_size = 16
lr = 0.01


fn = "cora-net"#"brazil-airports","europe-airports","usa-airports" ,
name = fn+".edgelist"
g = ig.Graph.Read_Ncol(name)
names = g.vs()["name"]
num_node = len(g.vs())


labs =  pd.read_csv("labels-"+fn+".txt",sep=" ")
labs = labs.set_index("node")
num_classes = len(labs["label"].unique())

x = np.load("cora_features.npy")
num_features = x.shape[1]
    

source = []
target = []
for i in g.es():
    source.append(i.tuple[0])
    target.append(i.tuple[1])
    
edge_index = torch.tensor([source,target])

#train_samples = [i for i in range(140)]#np.random.choice( [i for i in range(len(names))] ,  round(5*len(names)/100),replace=False) 

#------- Input to the extreme method
train_mask = torch.tensor([True if i<140 else False for i in range(len(g.vs()))])
val_mask = torch.tensor([True if (i>=140 and i<640) else False for i in range(len(g.vs()))])
test_mask = torch.tensor([True if (i>=1708) else False for i in range(len(g.vs()))])


#------- Define the input variables 
y = torch.tensor(labs["label"].values.tolist())
x = torch.tensor(x)    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------------------------ Regular
dat = Data(edge_index = edge_index, test_mask = test_mask, 
            train_mask = train_mask, val_mask = val_mask, x = x , y = y)    



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), dat.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)], lr=0.01)

n_epochs = 200
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    
    