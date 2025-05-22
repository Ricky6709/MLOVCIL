# 定义神经网络类
import torch
import torch.nn as nn
import torch.nn.functional as F

class cls_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        x = torch.matmul(adjacency_matrix, x)
        x = self.linear(x)
        x = F.relu(x)
        return x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, x, adjacency_matrix):
        x = self.gc1(x, adjacency_matrix)
        x = self.gc2(x, adjacency_matrix)
        return x