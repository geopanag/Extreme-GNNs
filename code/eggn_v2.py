#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:56:35 2019

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
from torch_geometric.nn import GCNConv
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


import igraph as ig
import scipy.io
import networkx as nx
import numpy as np

import torch_geometric.transforms as T

os.chdir("/home/george/Desktop/extreme-gnns/code")

           
class gcn(torch.nn.Module):
    """
    https://arxiv.org/abs/1609.02907    
    """
    #------- Regular
    def __init__(self):
        super(gcn, self).__init__()
        self.conv1 = GCNConv(dat.num_features, dat.hidden_size)
        self.conv2 = GCNConv(dat.hidden_size, dat.num_classes) #+1

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        
    def forward(self, dat):
        x, edge_index = dat.x, dat.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



def get_indices(samples,dic,thres):    
    extreme_names = [name  for name in samples  if dic[name]>=thres ] 
    idx = [names.index(i) for i in extreme_names]
    extremes  = np.repeat(False,len(names)) 
    extremes[idx] = True 
    
    regular_names = [name  for name in samples  if dic[name]<thres ] 
    idx = [names.index(i) for i in regular_names]
    regulars = np.repeat(False,len(names)) 
    regulars[idx] = True 
    
    #----- check the number of edges (e.g. the structure contained) in the extreme nodes compared to regular ones
    print("extreme edges:"+str(sum([dic[i] for i in extreme_names])))
    print("regular edges:"+str(sum([dic[i] for i in regular_names])))
    
    return extremes, regulars


def train(dat):   
    model.train()
    optimizer.zero_grad()
    out = model(dat)
    F.nll_loss(out[dat.train_mask], dat.y[dat.train_mask]).backward()
    optimizer.step()
    

def test(dat):
    model.eval()
    logits, accs = model(dat), []
    for _, mask in dat('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(dat.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
    

    
os.chdir("/home/george/Desktop/extreme-gnns/data")
           
if __name__ == '__main__': 
    #------- Dataset & Parameters
    hidden_size = 20
    lr = 0.01
    n_epochs = 300
    

    thresholds = eval(open('degree_thresholds.txt' , 'r').read())
            
    #------- Read the data
    fn = "cora-net" #for fn in []:#"brazil-airports","europe-airports","usa-airports" ,
    name = fn+".edgelist"
    g = ig.Graph.Read_Ncol(name)
    names = g.vs()["name"]
    num_node = len(g.vs())
    
    # degreee based
    degs = g.degree()
    dic = {i:j for i,j in zip(names,degs)}

    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath("/home/george/Desktop/extreme-gnns")), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, T.NormalizeFeatures())
    data = dataset[0]
    
    labs =  data.y.numpy()
    #labs = labs.set_index("node")
    num_classes = len(np.unique(labs))

    #x = np.load("cora_features.npy")
    #num_features = x.shape[1]
    thres = thresholds["cora-net"]
                
    extremes = [True  if i>thres else False for i in degs] 
    regs = [True  if i<=thres else False for i in degs] 
    
    train_samples = [names[i] for i in np.where(data.train_mask.numpy())[0]]#np.random.choice( [i for i in range(len(names))] ,  round(20*len(names)/100),replace=False) 
    
    test_samples = [names[i] for i in np.where(data.test_mask.numpy())[0]]
    del data.val_mask
    
    
    train_extremes, train_regulars =  get_indices(train_samples,dic,thres)
    test_extremes, test_regulars =  get_indices(test_samples,dic,thres)
        
    #------- Create the input to the algorithm
    #source = []
    #target = []
    #for i in g.es():
    #    source.append(i.tuple[0])
    #    target.append(i.tuple[1])
    #edge_index = torch.tensor([source,target])
    
    
    #------- Input to the extreme head
    train_mask_e = torch.tensor(train_extremes)
    test_mask_e =  torch.tensor(test_extremes)
    
    
    #--- Input to the regular head 
    train_mask_r = torch.tensor(train_regulars)
    test_mask_r =  torch.tensor(test_regulars)#[not i for i in test_extremes])
    
    #------- Define the input variables 
    #y = torch.tensor(labs["label"].values.tolist())
    #x = torch.tensor(x)    

    num_features = data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    #------------------------------ Regular
    dat_regular = Data(edge_index = data.edge_index, test_mask = test_mask_r, 
                train_mask = train_mask_r, x = data.x , y = data.y)    
    
    model, dat_regular = gcn(num_features,num_classes,hidden_size).to(device), dat_regular.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    print("regular")    
    best_test_acc = test_acc = 0
    for epoch in range(1, n_epochs+1):
        train(dat_regular)
        train_acc, tmp_test_acc = test(dat_regular)
        if tmp_test_acc > best_test_acc:
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f},  Test: {:.4f}'
        if epoch%50==0:
            print(log.format(epoch, train_acc, test_acc))
   
    
    #------------------------------ Extreme
    dat_extreme = Data(edge_index = data.edge_index, test_mask = test_mask_e, 
                train_mask = train_mask_e, x = data.x , y = data.y).to(device)   
    
    model = gcn(num_features,num_classes,hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    print("extreme")    
    best_test_acc = test_acc = 0
    for epoch in range(1, n_epochs+1):
        train(dat_extreme)
        train_acc, tmp_test_acc = test(dat_extreme)
        if tmp_test_acc > best_test_acc:
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f},  Test: {:.4f}'
        if epoch%50==0:
            print(log.format(epoch, train_acc, test_acc))
       
        
    
    #---------------------- Base Regular
    dat_baseline_reg = Data(edge_index = data.edge_index, test_mask = test_mask_r, 
                train_mask = data.train_mask, x = data.x , y = data.y).to(device)   
    
    model = gcn(num_features,num_classes,hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    
    print("baseline regular")
    best_test_acc = test_acc = 0
    for epoch in range(1, n_epochs+1):
        train(dat_baseline_reg)
        train_acc, tmp_test_acc = test(dat_baseline_reg)
        if tmp_test_acc > best_test_acc:
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f},  Test: {:.4f}'
        if epoch%50==0:
            print(log.format(epoch, train_acc, test_acc))

        
    #---------------------- Base Extreme
    dat_baseline_ex = Data(edge_index = data.edge_index, test_mask = test_mask_e, 
                train_mask = data.train_mask, x = data.x , y = data.y).to(device)   
        
    model = gcn(num_features,num_classes,hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
   
    print("baseline extreme")
    best_test_acc = test_acc = 0
    for epoch in range(1, n_epochs+1):
        train(dat_baseline_ex)
        train_acc, tmp_test_acc = test(dat_baseline_ex)
        if tmp_test_acc > best_test_acc:
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f},  Test: {:.4f}'
        if epoch%50==0:
            print(log.format(epoch, train_acc, test_acc))
    
    