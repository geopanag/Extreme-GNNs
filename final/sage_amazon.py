#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:53:58 2020

@author: george
"""

import os
import pandas as pd
import os.path as osp
import argparse

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon, Coauthor

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import matplotlib.pyplot as plt   

from sklearn.metrics import roc_auc_score
import igraph as ig
import scipy.io
import networkx as nx
import numpy as np

import torch_geometric.transforms as T

from collections import Counter
import operator




def find_thres_naive(deg_val,percentile):
    uni = np.unique(deg_val)
    perc = 0
    for i in uni:
        perc+=deg_val.count(i)
        if(perc*100/len(deg_val)>=percentile):
            break
    return i



class Net(torch.nn.Module):
    def __init__(self,num_features,num_classes,hidden_size):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(
            num_features,
            hidden_size)

        self.conv2 = SAGEConv(
            hidden_size,
            num_classes)
        self.reg_params=self.conv1.parameters()
	

    def forward(self,dat):
        x, edge_index = dat.x, dat.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



def train(dat):
    model.train()
    optimizer.zero_grad()
    out = model(dat)
    F.nll_loss(out[dat.train_mask], dat.y[dat.train_mask]).backward()
    optimizer.step()


def test(dat):
    model.eval()
    logits, accs = model(dat), []
    for _, mask in dat('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        pred = pred.detach().cpu().numpy()
        labels = dat.y[mask].detach().cpu().numpy()
        accs.append(roc_auc_score(labels,pred))
    return accs


def get_label(train_extremes,y):
    d = dict(Counter([y[i] for i in np.where(train_extremes)[0]]))
    return  max(d.items(), key=operator.itemgetter(1))[0]



def separate_indices(samples,dic,thres):
    # Separate the indices of the data that are extreme and regular
    
    idx_e = [name  for name in samples  if dic[name]>=thres ]
    extremes  = np.repeat(False,len(names))
    extremes[idx_e] = True

    idx_r = [name  for name in samples  if dic[name]<thres ]
    regulars = np.repeat(False,len(names))
    regulars[idx_r] = True

    #----- check the number of edges (e.g. the structure contained) in the extreme nodes compared to regular ones
    print("extreme nodes:"+str(len(idx_e)))
    print("regular nodes:"+str(len(idx_r)))
    print("extreme edges:"+str(sum([dic[i] for i in idx_e])))
    print("regular edges:"+str(sum([dic[i] for i in idx_r])))

    return extremes, regulars



if __name__ == '__main__':

    os.chdir("/data") 
    #------- Dataset & Parameters
    hidden_size = 32
    hidden_base = 64
    lr = 0.01
    n_epochs = 500
    experiments = 10 
    step_perc = 2
    repetitions = 10
    class_sample = 20

    for ri in range(repetitions):
        #https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
        np.random.seed(42+ri)
        for ds in  ['photo']:
            path = osp.join('data', ds)
        	dataset = Amazon(path,ds,T.NormalizeFeatures())
            
        	data = dataset[0]
        	edgelist = data.edge_index.numpy()
        
        	g = nx.DiGraph()
        	# because some nodes are not in the edgelists
        	g.add_nodes_from(range(max(max(edgelist[0]),max(edgelist[1]))))
        
        	for i in range(len(edgelist[0])):
        	    g.add_edge(edgelist[0][i],edgelist[1][i])
        
        	# degreee based
        	degs = dict(g.degree())
            logw = open("../results/logw_test_sage_"+str(ri)+"_"+ds+".csv","w")
            logw_val = open("../results/logw_val_sage_"+str(ri)+"_"+ds+".csv","w")
            logw.write("perc,test_r,test_br,test_e,test_be\n")
            logw_val.write("perc,val_r,val_br,val_e,val_be\n")
        	names = list(g.nodes())
        
        	labs =  data.y.numpy()
        	num_classes = len(np.unique(labs))
        
        	#-----------------------------------------------
        
        	percentile = 90
        	deg_val = list(degs.values())
        
        	test_set_size = len(names)/2
        	test_samples = np.random.choice([i for i in range(len(names))],  round(test_set_size),replace=False)
        	
        	y = data.y.numpy()
        	y = y[~test_samples]
        	
        	#---- 20 samples per class
        	train_samples = []
        	for cl in np.unique(y):
        	    train_samples.extend(np.random.choice(np.where(y==cl)[0],class_sample,replace =False))
        	 
        	val_samples =[idx for idx in range(len(y)) if idx not in train_samples ]
        	
        	train_all = np.repeat(False,len(data.y.numpy()))
        	train_all[train_samples] = True
        
        	for exp in range(experiments):
        	    # find the threshold
        	    thres = find_thres_naive(deg_val,percentile)
        	    
        	    percentile = percentile - step_perc
        	        
        	    print("train")
        	    train_extremes, train_regulars =  separate_indices(train_samples,degs,thres)
        	    print("val")
        	    val_extremes, val_regulars =  separate_indices(val_samples,degs,thres)
        	    print("test")
        	    test_extremes, test_regulars =  separate_indices(test_samples,degs,thres)
        	                
        	    #----- to binary
        	    y = data.y.numpy()
        	    extreme_train_label = get_label(train_extremes,y)
        	    data.y = torch.tensor(np.array([i==extreme_train_label for i in list(y)]).astype(int))
        
        	    #------- Input to the extreme head
        	    train_mask_e = torch.tensor(train_extremes)
        	    val_mask_e = torch.tensor(val_extremes)
        	    test_mask_e =  torch.tensor(test_extremes)
        
        
        	    #--- Input to the regular head 
        	    train_mask_r = torch.tensor(train_regulars)
        	    val_mask_r = torch.tensor(val_regulars)
        	    test_mask_r =  torch.tensor(test_regulars)#[not i for i in test_extremes])
        
        
        	    train_all = torch.tensor(train_all) 
        	    
        	    #------- Define the input variables 
        	    num_features = data.x.shape[1]
        	    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        	    #------------------------------ Regular
        	    dat_regular = Data(edge_index = data.edge_index,
        	                       test_mask = test_mask_r, train_mask = train_mask_r,
        	                       x = data.x , y = data.y, val_mask = val_mask_r).to(device)
        
        	    model = Net(num_features,num_classes,hidden_size).to(device)
        	    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        	    print("regular")
        	    best_test_r = 0
        	    best_val_acc_r = test_acc = 0
        	    for epoch in range(1, n_epochs+1):
        	        train(dat_regular)
        	        train_acc, val_acc, tmp_test_acc = test(dat_regular)
        	        if val_acc > best_val_acc_r:
        	            best_val_acc_r = val_acc
        	            test_acc = tmp_test_acc
        
        	        if(test_acc>best_test_r):
        	            best_test_r = test_acc
        	        if epoch%50==0:
        	            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        	            print(log.format(epoch, train_acc, best_val_acc_r, test_acc))
        
        
        	    #------------------------------ Extreme
        	    dat_extreme = Data(edge_index = data.edge_index, test_mask = test_mask_e,
        	                train_mask = train_mask_e, x = data.x , y = data.y, val_mask = val_mask_e).to(device)
        
        	    model = Net(num_features,num_classes,hidden_size).to(device)
        	    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        	    
        	    print("extreme")
        	    best_test_e = 0
        	    best_val_acc_e = test_acc = 0
        	    for epoch in range(1, n_epochs+1):
        	        train(dat_extreme)
        	        train_acc, val_acc, tmp_test_acc = test(dat_extreme)
        	        if val_acc > best_val_acc_e:
        	            best_val_acc_e = val_acc
        	            test_acc = tmp_test_acc
        
        	        if(test_acc>best_test_e):
        	            best_test_e = test_acc
        	        if epoch%50==0:
        	            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        	            print(log.format(epoch, train_acc, best_val_acc_e, test_acc))
        
        
        	    #---------------------- Base Regular
        	    dat_baseline_reg = Data(edge_index = data.edge_index, test_mask = test_mask_r,
        	                train_mask = train_all, x = data.x , y = data.y, val_mask = val_mask_r).to(device)
        
        	    model = Net(num_features,num_classes,hidden_base).to(device)
        	    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        
        	    print("baseline regular")
        	    best_test_br = 0
        	    best_val_acc_br = test_acc = 0
        	    for epoch in range(1, n_epochs+1):
        	        train(dat_baseline_reg)
        	        train_acc, val_acc, tmp_test_acc = test(dat_baseline_reg)
        	        if val_acc > best_val_acc_br:
        	            best_val_acc_br = val_acc
        	            test_acc = tmp_test_acc
        	        if(test_acc>best_test_br):
        	            best_test_br = test_acc
        	        if epoch%50==0:
        	            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        	            print(log.format(epoch, train_acc, best_val_acc_br, test_acc))
        
        	    #---------------------- Base Extreme
        	    dat_baseline_ex = Data(edge_index = data.edge_index, test_mask = test_mask_e,
        	                train_mask = train_all, x = data.x , y = data.y, val_mask = val_mask_e).to(device)
        
        	    model = Net(num_features,num_classes,hidden_base).to(device)
        	    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        	    print("baseline extreme")
        	    best_test_be = 0
        	    best_val_acc_be = test_acc = 0
        	    for epoch in range(1, n_epochs+1):
        	        train(dat_baseline_ex)
        	        train_acc, val_acc, tmp_test_acc = test(dat_baseline_ex)
        	        if val_acc > best_val_acc_be:
        	            best_val_acc_be = val_acc
        	            test_acc = tmp_test_acc
        	        if(test_acc>best_test_be):
        	            best_test_be = test_acc
        	        if epoch%50==0:
        	            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        	            print(log.format(epoch, train_acc, best_val_acc_be, test_acc))
        
        	    logw_val.write(str(percentile)+","+str(best_val_acc_r)+"," +str(best_val_acc_br)+","+str(best_val_acc_e)+","+str(best_val_acc_be)+"\n")
        	    logw.write(str(percentile)+","+str(best_test_r)+"," +str(best_test_br)+","+str(best_test_e)+","+str(best_test_be)+"\n")
        	
        	logw_val.close()    
        	logw.close()



