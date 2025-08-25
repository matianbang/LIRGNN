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
    read_directory_test = "/data/1.mitbih/mitbih_dataset_test_500.txt"

    with open(read_directory_test, "rb") as fp:  
        test_list = pickle.load(fp) 
    
    print('===========Graph Generator===========')
    print('Please wait a moment......')
    graphs = graph_generator(test_list)
    
    for i in range(43):
        match_set = graphs[50 * i:50 * (i+1)]
        for j in range(50):
            match_set[j].y = i

    test_loader = DataLoader(graphs, batch_size=128, shuffle=False)
            
    pretrained_path = './mitbih_model_GNN_Transform.pth'
    pre_data = torch.load(pretrained_path, map_location=lambda storage, loc:storage)
    pre_dict = pre_data['model']

    model = GraphOOD_Network()
    model_dict = model.state_dict()

    pre_dict = {k:v for k,v in pre_dict.items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    model = model.to(dev)
    model.eval()

    out_test_datasets = []
    test_labels = []
    
    print('===========start testing===========')
    for data in test_loader:
        data = data.to(dev)
        c_feature, c_graph_feat, s_graph_feat = model(data)
        out_test_datasets.append(c_feature)
        test_labels.append(data.y)

    out_test_datasets = torch.cat(out_test_datasets,dim=0)
    test_labels = torch.cat(test_labels)

    # 模板
    templates = torch.eye(43).to(dev)

     # 余弦相似性 
    score = []
    for i in range(len(out_test_datasets)):
        distances = []
        val_mat = out_test_datasets[i]
        for j in range(len(templates)):
            template_mat = templates[j]
            cos_sim = F.cosine_similarity(val_mat, template_mat, dim=0)
            distances.append(cos_sim)
        index = distances.index(max(distances))
        pred_label = index
        if test_labels[i] == pred_label:
            score.append(1)
        else:
            score.append(0)

    print(sum(score)/len(score))
          
if __name__ == '__main__':
    main()

