import torch
from torch.nn import Linear, Dropout, ReLU, ELU, Tanh, Softmax, Sequential, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, CGConv


## CGCNN model from Xie & Grossman
# class CGCNNet(torch.nn.Module):
#     def __init__(self, num_node_features, num_edge_features, num_classes,
#                  interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=8, batch_norm=True,
#                  concat=False, aggr="add"):
#         super(CGCNNet, self).__init__()
#         # self.convolution = Sequential(CGConv(channels=(num_node_features, interlayer_dim),
#         #                                      dim=num_edge_features),
#         #                               ReLU(),
#         #                               Dropout(drop_p),
#         #                               CGConv(channels=(interlayer_dim, interlayer_dim),
#         #                                      dim=num_edge_features),
#         #                               ReLU(),
#         #                               Dropout(drop_p),
#         #                               CGConv(channels=(interlayer_dim, output_dim),
#         #                                      dim=num_edge_features),
#         #                               BatchNorm1d(output_dim)
#         #                               )
#         self.convolution = CGConv(channels=num_node_features, 
#                                   dim=num_edge_features)
#         self.classifier = Linear(output_dim, num_classes)

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         x = self.convolution(x, edge_index, edge_attr)

#         return self.classifier(x)

## Basic Convolutional GNN (Kipf)
class GCNNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 256)
        self.classifier = Linear(256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return self.classifier(x)


## Attention network (Velikovic)
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, 128)
        self.conv2 = GATv2Conv(128, 128)
        self.conv3 = GATv2Conv(128, 256)
        self.classifier = Linear(256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        # x = F.relu(x)

        return self.classifier(x)
