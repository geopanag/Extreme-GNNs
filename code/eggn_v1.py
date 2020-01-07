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
    #------- Regular
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


def get_indices(samples,dic,thres):
    extreme_names = [name  for name in [names[i] for i in samples]  if dic[name]>=thres ] 
    idx = [names.index(i) for i in extreme_names]
    extremes  = np.repeat(False,len(names)) 
    extremes[idx] = True 
    
    regular_names = [name  for name in [names[i] for i in samples]  if dic[name]<thres ] 
    idx = [names.index(i) for i in regular_names]
    regulars = np.repeat(False,len(names)) 
    regulars[idx] = True 
    #------!!! check the number of edges (e.g. the structure contained) in the extreme nodes compared to regular ones !!!
    print("extreme edges:"+str(sum([dic[i] for i in extreme_names])))
    
    print("regular edges:"+str(sum([dic[i] for i in regular_names])))
    
    return extremes, regulars


def train(dat, model, lr, n_epochs):
        #------ Run training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        model.train()
    
        for epoch in range(n_epochs):
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
    
    
    
os.chdir("/home/george/Desktop/Extreme-GNNs/data")
           
if __name__ == '__main__': 
    #------- Dataset & Parameters
    hidden_size = 10
    
    x = np.load("cora_features.npy")
    num_features = x.shape[1]
    
    labs =  pd.read_csv("cora_node_labels.txt",header=None)
    labs = labs.set_index(0)
    num_classes = len(labs[1].unique())
    
    #d = pd.read_csv("cora_edges.txt",header=None)
    #d.to_csv("cora_net.txt",sep="\t",index=False,header=False)
    name = "cora_net.txt"
    g = ig.Graph.Read_Ncol(name)
    names = g.vs()["name"]
    num_node = len(g.vs())
    degs = g.degree()
    thres = 6.8399999999999981
    lr = 0.01
    n_epochs = 200
    
    dic = {i:j for i,j in zip(names,degs)}
    
    
    #extremes = [True  if i>cora_thres else False for i in degs] 
    
    
    #------ Sample random 20% for train set
    train_samples = np.random.choice( [i for i in range(len(names))] ,  round(20*len(names)/100),replace=False) 
    
    test_samples = [i for i in range(len(names)) if i not in train_samples] 
    
    
    train_extremes, train_regulars =  get_indices(train_samples,dic,thres)
    print("-----------")
    test_extremes, test_regulars =  get_indices(test_samples,dic,thres)
    
    
    #ng = g.subgraph(extremes_names)
    #len([i for i in ng.es()])
    
    #ng = g.subgraph(regular_names)
    #len([i for i in ng.es()])
    
    #------- Create the input to the algorithm
    source = []
    target = []
    for i in g.es():
        source.append(i.tuple[0])
        target.append(i.tuple[1])
        
    edge_index = torch.tensor([source,target])
    
    
    #------- Input to the extreme method
    #train_idx_e = extremes  #np.random.choice(num_node, round(10*num_node/100) ) 
    #train_idx_r = np.random.choice(num_node, round(10*num_node/100) ) 
    train_mask_e = torch.tensor(train_extremes)
    test_mask_e =  torch.tensor(test_extremes)
    
    
    #--- random 
    #train_idx = np.random.choice(np.where(regs)[0], round(10*sum(regs)/100) ) 
    #train_mask_r = torch.tensor([True if i in train_idx else False for i in range(0,num_node)] )
    #test_mask_r =  torch.tensor([True if i not in train_idx else False for i in range(0,num_node)] )
    train_mask_r = torch.tensor(train_regulars)
    test_mask_r =  torch.tensor(test_regulars)#[not i for i in test_extremes])
    
    #------- Define the input of the dataset
    #y = torch.FloatTensor([np.random.choice(range(0,num_classes),sum(train_mask))])
    y = torch.tensor(labs[1].values.tolist())
    x = torch.tensor(x)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #------------------------------ Regular
    dat = Data(edge_index = edge_index, test_mask = test_mask_r, 
                train_mask = train_mask_r, x = x , y = y)    
    
    rgnn = gnn(num_features,num_classes,hidden_size)
    model = rgnn.to(device)
   
    print("regular")
    train(dat, model, lr, n_epochs)
    
    
    #------------------------------ Extreme
    dat = Data(edge_index = edge_index, test_mask = test_mask_e, 
                train_mask = train_mask_e, x = x , y = y)    
    
    #------ Set labels
    egnn = gnn(num_features,num_classes,hidden_size)
    model = egnn.to(device)
    
    print("extreme")
    train(dat, model, lr, n_epochs)
        

    #------------------------------ Baseline
    full_train = [False for i in range(len(names))] 
    for i in train_samples:
        full_train[i] = True
    
    sum(test_regulars)
    sum(test_extremes)
    sum(full_train)
    sum(test_regulars+test_extremes+full_train)
     
    
    train_mask = torch.tensor(full_train)
    test_mask_r =  torch.tensor(test_regulars)
    test_mask_e =  torch.tensor(test_extremes)
    
    dat = Data(edge_index = edge_index, test_mask = test_mask_r, 
                train_mask = train_mask, x = x , y = y)    
    
    rgnn = gnn(num_features,num_classes,hidden_size)
    model = rgnn.to(device)

    print("baseline regular")
    train(dat, model, lr, n_epochs)
    
    #----------------------    
    dat = Data(edge_index = edge_index, test_mask = test_mask_e, 
                train_mask = train_mask, x = x , y = y)    
        
    egnn = gnn(num_features,num_classes,hidden_size)
    model = egnn.to(device)

    print("baseline extreme")
    train(dat, model, lr, n_epochs)
    
    
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


                                

