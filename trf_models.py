## Transformer-based network with edges attributes for parameters optimization and fine tuning
import torch
from torch.nn import Linear, Dropout, ReLU, ELU, Tanh, Softmax, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv


## Transformer-based network with edges attributes
class TrfEdgeNet(torch.nn.Module):
    ## Default parameters correspond to the best architecture according to optimization
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=8, batch_norm=True,
                 concat=False, aggr="add"):
        super(TrfEdgeNet, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=interlayer_dim,
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf2 = TransformerConv(in_channels=interlayer_dim, out_channels=interlayer_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.trf3 = TransformerConv(in_channels=interlayer_dim, out_channels=output_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.classifier = Linear(output_dim, num_classes)
        self.__activation = {"relu": ReLU(),
                             "elu": ELU(),
                             "tanh": Tanh(),
                             "softmax": Softmax(dim=1)}[activation]
        self.__drop_p = drop_p
        self.do_norm = batch_norm
        self.normalization = BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.trf1(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf2(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf3(x, edge_index, edge_attr)
        #x = self.__activation(x)
        #x = F.dropout(x, p=self.__drop_p, training=self.training)
        if self.do_norm:
            x = self.normalization(x)

        return self.classifier(x)
    

class TrfEdgeNetNoActDrop(torch.nn.Module):
    ## Default parameters correspond to the best architecture according to optimization
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=8, batch_norm=True,
                 concat=False, aggr="add"):
        super(TrfEdgeNetNoActDrop, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=interlayer_dim,
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf2 = TransformerConv(in_channels=interlayer_dim, out_channels=interlayer_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.trf3 = TransformerConv(in_channels=interlayer_dim, out_channels=output_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.classifier = Linear(output_dim, num_classes)
        self.__activation = {"relu": ReLU(),
                             "elu": ELU(),
                             "tanh": Tanh(),
                             "softmax": Softmax(dim=1)}[activation]
        self.__drop_p = drop_p
        self.do_norm = batch_norm
        self.normalization = BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.trf1(x, edge_index, edge_attr)
        #x = self.__activation(x)
        #x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf2(x, edge_index, edge_attr)
        #x = self.__activation(x)
        #x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf3(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        if self.do_norm:
            x = self.normalization(x)

        return self.classifier(x)
    
    
class TrfEdgeElemNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, 
                 elements={},
                 activation="relu", drop_p=0.3, heads=8, batch_norm=True,
                 concat=False, aggr="add"):
        super(TrfEdgeElemNet, self).__init__()
        
        self.elements = elements
        if self.elements:
            self.conv ={e: TrfEdgeNet(num_node_features, num_edge_features, num_classes).to(torch.device("cuda")) 
                        for e in self.elements}
        else:
            self.conv = TrfEdgeNet(num_node_features, num_edge_features, num_classes).to(torch.device("cuda"))
        self._activation = ReLU()
        self._dropout=Dropout(drop_p)
        self.batch_norm = BatchNorm1d(output_dim)
        self.classifier = Linear(output_dim, num_classes)

    def forward(self, data):
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        #data = data.to(torch.device("cuda"))
        #out = torch.tensor([]).to(torch.device("cuda"))
        out = []
        for el in self.elements:
            out.append(self.conv[el](data))
            
        return torch.flatten(torch.stack(out), end_dim=1)
    
    
class TrfEdgeNetL1(torch.nn.Module):
    ## Default parameters correspond to the best architecture according to optimization
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=8, batch_norm=True,
                 concat=False, aggr="add"):
        super(TrfEdgeNetL1, self).__init__()
        self.trf = TransformerConv(in_channels=num_node_features, out_channels=output_dim,
                                   edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.classifier = Linear(output_dim, num_classes)
        self.__activation = {"relu": ReLU(),
                             "elu": ELU(),
                             "tanh": Tanh(),
                             "softmax": Softmax(dim=1)}[activation]
        self.__drop_p = drop_p
        self.do_norm = batch_norm
        self.normalization = BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.trf(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        #if self.do_norm:
        x = self.normalization(x)

        return self.classifier(x)


## Transformer-based network with edges attributes, 5 layers
class TrfEdgeNetL5(torch.nn.Module):
    ## Default parameters correspond to the best architecture according to optimization
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim_1=200, interlayer_dim_2=300, output_dim=400,
                 activation="relu", drop_p=0.3, heads=8, batch_norm=True, concat=False, aggr="add"):
        super(TrfEdgeNetL5, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=interlayer_dim_1,
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf2 = TransformerConv(in_channels=interlayer_dim_1, out_channels=interlayer_dim_1,
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf3 = TransformerConv(in_channels=interlayer_dim_1, out_channels=interlayer_dim_2,
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf4 = TransformerConv(in_channels=interlayer_dim_2, out_channels=interlayer_dim_2, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)
        self.trf5 = TransformerConv(in_channels=interlayer_dim_2, out_channels=output_dim, edge_dim=num_edge_features,
                                    heads=heads, concat=concat, aggr=aggr)

        self.classifier = Linear(output_dim, num_classes)

        self.__activation = {"relu": ReLU(),
                             "elu": ELU(),
                             "tanh": Tanh(),
                             "softmax": Softmax(dim=1)}[activation]
        self.__drop_p = drop_p
        self.do_norm = batch_norm
        self.normalization = BatchNorm1d(output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.trf1(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf2(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf3(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf4(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf5(x, edge_index, edge_attr)
        if self.do_norm:
            x = self.normalization(x)

        return self.classifier(x)


## Transformer-based network with edges attributes + concatenation of heads
class TrfEdgeNetC(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=1, batch_norm=False,
                 concat=True, aggr="add"):
        super(TrfEdgeNetC, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=int(output_dim / 2 / heads),
                                    edge_dim=num_edge_features, heads=heads, concat=concat, aggr=aggr)
        self.trf2 = TransformerConv(in_channels=int(output_dim / 2 / heads) * heads,
                                    out_channels=int(output_dim / 2 / heads), edge_dim=num_edge_features, heads=heads,
                                    concat=concat, aggr=aggr)
        self.trf3 = TransformerConv(in_channels=int(output_dim / 2 / heads) * heads,
                                    out_channels=int(output_dim / heads), edge_dim=num_edge_features, heads=heads,
                                    concat=concat, aggr=aggr)
        self.classifier = Linear(int(output_dim / heads) * heads, num_classes)
        self.__activation = {"relu": ReLU(),
                             "elu": ELU(),
                             "tanh": Tanh(),
                             "softmax": Softmax(dim=1)}[activation]
        self.__drop_p = drop_p
        self.do_norm = batch_norm
        self.normalization = BatchNorm1d(int(output_dim / heads) * heads)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.trf1(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf2(x, edge_index, edge_attr)
        x = self.__activation(x)
        x = F.dropout(x, p=self.__drop_p, training=self.training)
        x = self.trf3(x, edge_index, edge_attr)
        if self.do_norm:
            x = self.normalization(x)

        return self.classifier(x)


class TrfEdgeNetRand(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(TrfEdgeNetRand, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=128, edge_dim=num_edge_features)
        self.trf2 = TransformerConv(in_channels=128, out_channels=128, edge_dim=num_edge_features)
        self.trf3 = TransformerConv(in_channels=128, out_channels=256, edge_dim=num_edge_features)
        self.classifier = Linear(256, num_classes)

        # layers = [self.trf1, self.trf2, self.trf3]
        # for l in layers:
        #     torch.nn.init.normal_(l.lin_key.weight)
        #     torch.nn.init.normal_(l.lin_query.weight)
        #     torch.nn.init.normal_(l.lin_value.weight)
        #     torch.nn.init.normal_(l.lin_edge.weight)
        # torch.nn.init.normal_(self.classifier.weight)
        # torch.nn.init.normal_(self.classifier.bias)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.trf1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.trf2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.trf3(x, edge_index, edge_attr)

        return self.classifier(x)


## Transformer-based network without edges attributes
class TrfNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(TrfNet, self).__init__()
        self.trf1 = TransformerConv(in_channels=num_node_features, out_channels=128)
        self.trf2 = TransformerConv(128, 128)
        self.trf3 = TransformerConv(128, 256)
        self.classifier = Linear(256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.trf1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.trf2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.trf3(x, edge_index)

        return self.classifier(x)
