# -*- coding: utf-8 -*-
import pdb
import time
import numpy as np
import pandas as pd
import pickle
import os
import torch
import random
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset,TensorDataset 
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from collections import defaultdict 
from model import GraphOOD_Network,graph_generator

     
def main():
    
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    print("CUDA available? ", torch.cuda.is_available())

    dev = torch.device('cuda:3') if torch.cuda.is_available() else torch.device("cpu")
    print("Device in use is ", dev)

    # 读取数据
    read_directory_train = "/data/1.mitbih/mitbih_dataset_train_500_.txt"

    with open(read_directory_train, "rb") as fp:  
        train_list = pickle.load(fp) 
        
    print('===========Graph Generator===========')
    print('Please wait a moment......')
    
    graphs = graph_generator(train_list)
    
    for i in range(43):
        match_set = graphs[450 * i:450 * (i+1)]
        for j in range(450):
            match_set[j].y = i

    random.shuffle(graphs)
    
    train_loader = DataLoader(graphs, batch_size=128, shuffle=True)

    # model
    model = GraphOOD_Network()
    model = model.to(dev)
    # 定义损失函数
    Loss = torch.nn.CrossEntropyLoss()
    MSELoss = torch.nn.MSELoss()
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 ,weight_decay=1e-5)

    print('===========start training===========')
    min_loss = 10000000000
    for epoch in range(1,101):
        model.train()
        epoch_loss = []
        for data in train_loader:
            
            data = data.to(dev)
            
            c_feature, c_graph_feat, s_graph_feat = model(data)

            inv_loss = Loss(c_feature, data.y)
            pdb.set_trace()
            mix_f_1 = model.mix_cs_1(c_graph_feat, s_graph_feat)
            mix_f_2 = model.mix_cs_2(c_graph_feat, s_graph_feat)

            cls_loss = Loss(mix_f_1, data.y) + Loss(mix_f_2, data.y)
            
            loss = cls_loss + inv_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
        print("epoch:",epoch, "Loss:",sum(epoch_loss))
        
        if sum(epoch_loss) < min_loss:
            min_loss = sum(epoch_loss)
            all_data = dict(optimizer = optimizer.state_dict(),
                    model = model.state_dict(),
                    epoch = epoch)
            file_name = './' + 'mitbih_model_GNN_Transform.pth'
            torch.save(all_data, file_name)    
    print("min_loss:",min_loss) 
          
if __name__ == '__main__':
    main()
    
       

