"""
reated on Tue Nov  5 18:56:35 2019

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


import igraph as ig
import scipy.io
import networkx as nx
import numpy as np

import torch_geometric.transforms as T

from collections import Counter
import operator




George = True #if True, George is running the code, otherwise, Hamid is working on the code

def order(X):
    """Return the order statistic of each sample in X, features by features
    """
    n, d = np.shape(X)
    R = np.sort(X, axis=0)
    return R


def transform_(deg_val):
    # take as input the list of degrees of the graph
    uni = np.unique(deg_val)
    table = np.zeros([len(uni),2])
    counter = 0
    for i in uni:
        table[counter] = [i,deg_val.count(i)]
        counter+=1
    
    #the original transform function
    R = order(table[:,1].reshape(-1,1)) 
    x = table[:,1].reshape(-1,1)
    
    n, d = np.shape(x)
    n_R = np.shape(R)[0]
    a = np.zeros((n, d))
    for i in range(d):
        a[:, i] = np.searchsorted(R[:, i], x[:, i]) / float(n_R + 1)
    return 1. / (1-a)



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
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self,dat):
        x = F.dropout(dat.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, dat.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, dat.edge_index)
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
        acc = pred.eq(dat.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs  


def get_label(train_extremes,y):
    d = dict(Counter([y[i] for i in np.where(train_extremes)[0]]))
    return  max(d.items(), key=operator.itemgetter(1))[0]

   
def separate_indices(samples,dic,thres):    
    # Separate the indices of the data that are extreme and regular
    idx_e = [name  for name in samples  if dic[name]>=thres ] 
    #idx = [names.index(i) for i in extreme_names]
    extremes  = np.repeat(False,len(names)) 
    extremes[idx_e] = True 
    
    idx_r = [name  for name in samples  if dic[name]<thres ] 
    #idx = [names.index(i) for i in regular_names]
    regulars = np.repeat(False,len(names)) 
    regulars[idx_r] = True 
    
    #----- check the number of edges (e.g. the structure contained) in the extreme nodes compared to regular ones
    print("extreme nodes:"+str(len(idx_e)))
    print("regular nodes:"+str(len(idx_r)))
    print("extreme edges:"+str(sum([dic[i] for i in idx_e])))
    print("regular edges:"+str(sum([dic[i] for i in idx_r])))
    
    return extremes, regulars
 
    


if George:
    os.chdir("/home/george/Desktop/extreme-gnns/data") 
   # os.chdir("/home/dascim/panago/data")
else:
    os.chdir("/home/h/Documents/Japet/Extreme-GNNs/data")
           
if __name__ == '__main__': 
    #------- Dataset & Parameters
    hidden_size = 32
    hidden_base = 64
    lr = 0.01
    n_epochs = 500
    experiments = 5
    step_perc = 4
        
    for ds in  ['CiteSeer','Cora','PubMed']:
        if George:
            path = osp.join(osp.dirname(
                    osp.realpath("/home/george/Desktop/extreme-gnns/data")), 'data', ds)
        else:
            path = osp.join(osp.dirname(osp.realpath("/home/h/Documents/Japet/Extreme-GNNs/")), '..', 'data', ds)
            
        dataset = Planetoid(path, ds, T.NormalizeFeatures())
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
        
        labs =  data.y.numpy()
        #labs = labs.set_index("node")
        num_classes = len(np.unique(labs))
        
        
        #-----------------------------------------------
        logw = open("../results_new/logw_egat_"+ds+".csv","w")
        logw.write("perc,test_r,test_br,test_e,test_be\n")
        # change the percentile of threshold for 5%
        
        percentile = 90
        deg_val = list(degs.values())
        
        for i in range(experiments):
            
            # find the threshold
            #V = transform_(deg_val)
            #thres = np.percentile(V, q=percentile)
            #print(thres)
            thres = find_thres_naive(deg_val,percentile)
            
            percentile = percentile-step_perc
            
            train_samples = [i for i in np.where(data.train_mask.numpy())[0] ] #np.random.choice( [i for i in range(len(names))] ,  round(20*len(names)/100),replace=False) 
            val_samples = [names[i] for i in np.where(data.val_mask.numpy())[0] ]     #np.random.choice( [i for i in range(len(names))] ,  round(20*len(names)/100),replace=False) 
            test_samples = [names[i] for i in np.where(data.test_mask.numpy())[0] ]
            #idx = [names.index(i) for i in extreme_names]
    
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
            
            #------- Define the input variables 
            num_features = data.x.shape[1]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


            #------------------------------ Regular
            dat_regular = Data(edge_index = data.edge_index, test_mask = test_mask_r, 
                        train_mask = train_mask_r, x = data.x , y = data.y, val_mask = val_mask_r).to(device)    
            
            model = Net(num_features,num_classes,hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            print("regular")    
            best_test_r = 0
            best_val_acc = test_acc = 0
            for epoch in range(1, n_epochs+1):
                train(dat_regular)
                train_acc, val_acc, tmp_test_acc = test(dat_regular)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    
                if(test_acc>best_test_r):
                    best_test_r = test_acc   
                if epoch%50==0:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, train_acc, best_val_acc, test_acc))
                
                
            #------------------------------ Extreme
            dat_extreme = Data(edge_index = data.edge_index, test_mask = test_mask_e, 
                        train_mask = train_mask_e, x = data.x , y = data.y, val_mask = val_mask_e).to(device)   
            
            model = Net(num_features,num_classes,hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            
            print("extreme")    
            best_test_e = 0
            best_val_acc = test_acc = 0
            for epoch in range(1, n_epochs+1):
                train(dat_extreme)
                train_acc, val_acc, tmp_test_acc = test(dat_extreme)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    
                if(test_acc>best_test_e):
                    best_test_e = test_acc   
                if epoch%50==0:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, train_acc, best_val_acc, test_acc))   
                
            
            #---------------------- Base Regular
            dat_baseline_reg = Data(edge_index = data.edge_index, test_mask = test_mask_r, 
                        train_mask = data.train_mask, x = data.x , y = data.y, val_mask = val_mask_r).to(device)   
            
            model = Net(num_features,num_classes,hidden_base).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            
            print("baseline regular")
            best_test_br = 0
            best_val_acc = test_acc = 0
            for epoch in range(1, n_epochs+1):
                train(dat_baseline_reg)
                train_acc, val_acc, tmp_test_acc = test(dat_baseline_reg)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                if(test_acc>best_test_br):
                    best_test_br = test_acc   
                if epoch%50==0:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, train_acc, best_val_acc, test_acc))  
        
            #---------------------- Base Extreme
            dat_baseline_ex = Data(edge_index = data.edge_index, test_mask = test_mask_e, 
                        train_mask = data.train_mask, x = data.x , y = data.y, val_mask = val_mask_e).to(device)   
                
            model = Net(num_features,num_classes,hidden_base).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
           
            print("baseline extreme")
            best_test_be = 0
            best_val_acc = test_acc = 0
            for epoch in range(1, n_epochs+1):
                train(dat_baseline_ex)
                train_acc, val_acc, tmp_test_acc = test(dat_baseline_ex)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                if(test_acc>best_test_be):
                    best_test_be = test_acc                
                if epoch%50==0:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch, train_acc, best_val_acc, test_acc))  
            
            logw.write(str(percentile)+","+str(best_test_r)+"," +str(best_test_br)+","+str(best_test_e)+","+str(best_test_be)+"\n")
            
    logw.close()

