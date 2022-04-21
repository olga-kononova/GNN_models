from pprint import pprint
from typing import Dict, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ReLU, Dropout

from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch_sparse import SparseTensor

from inspect import Parameter

#from features import ATOM_SYMBOLS
#num_sym_dict = {n:a for a, n in ATOM_SYMBOLS.items()}

class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.

    Args:
        channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        batch_norm (bool, optional): If set to :obj:`True`, will make use of
            batch normalization. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V_t}|, F_{t})` if bipartite
    """
    def __init__(self, 
                 in_channels: Union[int, Tuple[int, int]], 
                 out_channels: Union[int, Tuple[int, int]],
                 dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = False,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.batch_norm = batch_norm

        # if isinstance(channels, int):
        #     channels = (channels, channels)

        self.lin_f = Linear(in_channels*2 + dim, out_channels, bias=bias)
        self.lin_s = Linear(in_channels*2 + dim, out_channels, bias=bias)
        self.lin_t = Linear(in_channels, out_channels, bias=bias)
        if batch_norm:
            self.bn = BatchNorm1d(out_channels)
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()


    def forward(self, 
                x: Union[Tensor, PairTensor], 
                edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        #print("Out")
        #print(out.size())
        #print("x")
        #print(x[0].size())
        out += self.lin_t(x[1])
        return out


    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels, self.out_channels}, dim={self.dim})'
    

class CGCNNet(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=8, batch_norm=True,
                 concat=False, aggr="add"):
        super(CGCNNet, self).__init__()
        self.conv1 = CGConv(in_channels=num_node_features, 
                                             out_channels=interlayer_dim,
                                             dim=num_edge_features)
        self.conv2 = CGConv(in_channels=interlayer_dim, 
                            out_channels=interlayer_dim,
                            dim=num_edge_features)
        self.conv3 = CGConv(in_channels=interlayer_dim, 
                            out_channels=output_dim,
                            dim=num_edge_features)
        self._activation = ReLU()
        self._dropout=Dropout(drop_p)
        self.batch_norm = BatchNorm1d(output_dim)
        self.classifier = Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self._activation(x)
        x = self._dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self._activation(x)
        x = self._dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm(x)

        return self.classifier(x)
    
class CGCNNetL1(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes,
                 interlayer_dim=128, output_dim=256, activation="relu", drop_p=0.3, heads=8, batch_norm=True,
                 concat=False, aggr="add"):
        super(CGCNNetL1, self).__init__()
        self.conv = CGConv(in_channels=num_node_features, 
                            out_channels=output_dim,
                            dim=num_edge_features)

        self._activation = ReLU()
        self._dropout=Dropout(drop_p)
        self.batch_norm = BatchNorm1d(output_dim)
        self.classifier = Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv(x, edge_index, edge_attr)
        x = self._activation(x)
        x = self._dropout(x)
        x = self.batch_norm(x)

        return self.classifier(x)

    
