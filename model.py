import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models
from torch_geometric.utils.convert import from_networkx
from ts2vg import NaturalVG, HorizontalVG
import networkx as nx
from torch_geometric.nn import GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, degree, add_self_loops, batched_negative_sampling
from torch_scatter import scatter_add, scatter_mean
from collections import defaultdict
import numpy as np
import random

import pdb
 
def graph_generator(pulse_list):
    graphs = []
    for subject in range(len(pulse_list)):    
        for pulse in range(len(pulse_list[0])): 
            g = NaturalVG()
            g.build(pulse_list[subject][pulse]) 
            nx_g = g.as_networkx() 
            for i in range(nx_g.number_of_nodes()): 
                if i < 59 :
                    x1 = pulse_list[subject][pulse][i+1] - pulse_list[subject][pulse][i]
                if i < 58 :
                    x2 = (pulse_list[subject][pulse][i+2] - pulse_list[subject][pulse][i])/2
                if i < 57 :
                    x3 = (pulse_list[subject][pulse][i+3] - pulse_list[subject][pulse][i])/3
                if i < 56:
                    x4 = (pulse_list[subject][pulse][i+4] - pulse_list[subject][pulse][i])/4
                if i < 55:
                    x5 = (pulse_list[subject][pulse][i+5] - pulse_list[subject][pulse][i])/5
                
                nx_g.add_node(i, x=[pulse_list[subject][pulse][i],x1,x2,x3,x4,x5])    
            H = nx.Graph() # 创建一个没有边（edge）和节点（node）的空图
            H.add_nodes_from(sorted(nx_g.nodes(data=True))) # 从可以迭代容器中添加多个点
            H.add_edges_from(nx_g.edges(data=True)) # 只添加边，不包含属性值
            H.nodes()
            pyg_graph = from_networkx(H)  # 这个函数就是把图给可视化了。将边和节点都转为tensor张量了。边变为二维矩阵
            graphs += [pyg_graph]

    return graphs

class AttentionModule(nn.Module):
    def __init__(self, channel=64, reduction=8):  
        super(AttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
             
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc1(y)
        y = y.view(b, c, 1)
        out = x * y
        return out 

class Score_NetWork(nn.Module):
    def __init__(self,):
        super(Score_NetWork, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=28, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3) 
        
        self.attention = AttentionModule()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.linear = nn.Linear(64,64)

    def forward(self, x):
        #pdb.set_trace()
        x = x.reshape(-1,1,60)
        
        x1 = self.conv1(x)  #torch.Size([128, 64, 28])
        x2 = self.conv2(x) 
        x3 = self.conv3(x) 
        
        x_sum = x1 + x2 + x3 
        
        # Bi-directional Transformer
        x_out1 = self.transformer_encoder(x_sum)
        x_out2 = self.transformer_encoder(torch.flip(x_sum,[2]))
        x_out = x_out1 + x_out2  #torch.Size([128, 64, 84])

        x_out = self.attention(x_out)
        
        x_out = self.pool(x_out)
        x_out = x_out.reshape(-1,64)
        x_out = self.linear(x_out)
        score = x_out.sigmoid()
        
        return score

class Feature_Split_Network(nn.Module):
    def __init__(self,hidden_channels):
        super(Feature_Split_Network, self).__init__()  
        self.conv1 = GINConv(
            Sequential(Linear(6, hidden_channels),BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels*2), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels*2, hidden_channels*2), BatchNorm1d(hidden_channels*2), ReLU(),
                       Linear(hidden_channels*2, hidden_channels*4), ReLU()))
        
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels*4, hidden_channels*4), BatchNorm1d(hidden_channels*4), ReLU(),
                       Linear(hidden_channels*4, hidden_channels*8), ReLU())) 
        
        self.conv4 = GINConv(
            Sequential(Linear(hidden_channels*8, hidden_channels*8), BatchNorm1d(hidden_channels*8), ReLU(),
                       Linear(hidden_channels*8, hidden_channels*16), ReLU())) 

        self.linear = Linear(hidden_channels * 16, 43)
        

    def forward(self, graph_x, edge_index, batch, score): 
        #pdb.set_trace()
        h1 = self.conv1(graph_x, edge_index)  
        h2 = self.conv2(h1, edge_index) 
        h3 = self.conv3(h2, edge_index)  
        h4 = self.conv4(h3, edge_index)  

        graph_feat = global_mean_pool(h4, batch) # torch.Size([128, 64])
        
        c_graph_feat = graph_feat * score      #torch.Size([128, 64])
        s_graph_feat = graph_feat * (1 - score)  #torch.Size([128, 64])

        c_feature = self.linear(c_graph_feat)   #torch.Size([128, 43])
        
        return c_feature, c_graph_feat, s_graph_feat
    
class GraphOOD_Network(nn.Module):
    def __init__(self,):
        super(GraphOOD_Network, self).__init__()   
        self.score_network = Score_NetWork()
        self.feature_split_network = Feature_Split_Network(hidden_channels=4) 
        
        self.mix_proj = nn.Sequential(nn.Linear(64 * 2, 64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(),nn.Linear(64, 43))

    def mix_cs_1(self, c_f: torch.Tensor, s_f: torch.Tensor):
        n = c_f.size(0)
        perm = np.random.permutation(n)
        mix_f = torch.cat([c_f, s_f[perm]], dim=-1)
        proj_mix_f = self.mix_proj(mix_f)
        return proj_mix_f  
    
    def mix_cs_2(self, c_f: torch.Tensor, s_f: torch.Tensor):
        mix_f = torch.cat([c_f, s_f], dim=-1)
        proj_mix_f = self.mix_proj(mix_f)
        return proj_mix_f   

    def forward(self, data):

        signal_x = data.x[:,0].float()
        
        graphs_x = data.x.float()
        graphs_edge_index = data.edge_index
        graphs_batch = data.batch

        score = self.score_network(signal_x)   
        
        c_feature, c_graph_feat, s_graph_feat = self.feature_split_network(graphs_x, graphs_edge_index, graphs_batch, score)

        return c_feature, c_graph_feat, s_graph_feat
        
        
        

    
           
    
 