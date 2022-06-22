"""
BEWARE OF HARDCODED PARAMETERS
"""
import os
import json
import numpy as np

from tqdm import tqdm

import torch
from torch.nn import Linear, CrossEntropyLoss
import torch.nn.functional as F

from torch_geometric.data import Data


class GNNWrapper(torch.nn.Module):
    """
    Wrapper class for any graph NN with >=1 convolutional layer and only node features
    Args:
        model       :instance of pytorch geometric GNN model with node attributes only
        model_path  :path to trained model file
        device      :cpu or gpu
    """
    def __init__(self, model, model_path, name="", suffix="", device="cuda", noActDrop=False):
        super(GNNWrapper, self).__init__()
        self.model = model.to(device) if device == "cuda" else model
        self.model_path = model_path
        self.name = name
        self.suffix = suffix
        
        self.reduced = noActDrop
        self.dropp = 0.3

        self.device = device
        self.loss_fn = CrossEntropyLoss()

        self.conv_layers = []
        self.classifier = None
        self.activation = None
        self.normalization = None

    @property
    def __name__(self):
        return "{}_{}".format(self.model.__class__.__name__, self.suffix)

    def load_model(self):
        model_path = os.path.join(self.model_path, "{}_{}.model".format(self.name, self.suffix))
        print("Loading model", model_path)
        assert os.path.exists(self.model_path)
        assert os.path.exists(model_path)
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=torch.device(self.device)))
        self.split_layers()

    def split_layers(self):
        if self.model:
            for layer in self.model.children():
                if any(c in str(layer) for c in {"TransformerConv", "CGConv"}):
                    self.conv_layers.append(layer)
                if "Linear" in str(layer):
                    self.classifier = layer
                if "ReLU" in str(layer):
                    self.activation = layer
                if "BatchNorm" in str(layer):
                    self.normalization = layer

    def get_embeddings(self, torch_data):
        torch_data = torch_data.to(self.device) if self.device == "cuda" else torch_data
        convolution = F.dropout(self.activation(self.conv_layers[0](torch_data.x,
                                                torch_data.edge_index)), 
                                                p=self.dropp, training=False)
        for layer in self.conv_layers[1:-1]:
            convolution = F.dropout(self.activation(layer(convolution,
                                       torch_data.edge_index)), p=self.dropp, training=False)
        convolution = self.conv_layers[-1](convolution, torch_data.edge_index)
        return self.normalization(convolution)

    def predict(self, torch_data):
        embeddings = self.get_embeddings(torch_data)
        prediction = F.softmax(self.classifier(embeddings), dim=1)
        return prediction.argmax(1), embeddings, prediction

    def forward(self, x, edge_attr, edge_index):
        torch_data = Data(x=x,
                          edge_attr=edge_attr,
                          edge_index=edge_index)
        return self.predict(torch_data)


    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        test_loss, correct_per_mol, correct, total_atoms = 0, 0, 0, 0
        for data_batch in tqdm(loader):
            dev_data = data_batch.to(self.device)
            y = dev_data.y
            pred = self.model(dev_data)
            test_loss += self.loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).sum().item()
            total_atoms += y.size(0)
            correct_per_mol += int(pred.argmax(1).eq(y).all())
                
        test_loss /= len(loader)
        correct_per_mol /= len(loader)
        correct /= total_atoms
    
        return test_loss, correct, correct_per_mol

    @staticmethod
    def features_to_torch_vec(data_vec):
        """
        :param data_vec: dict{x, y, edge_attr, edge_index}
        :return:
        """
        if "y" in data_vec:
            return Data(x=torch.tensor(data_vec["x"], dtype=torch.float),
                        y=torch.LongTensor(data_vec["y"]),
                        edge_attr=torch.tensor(data_vec["edge_attr"], dtype=torch.float),
                        edge_index=torch.LongTensor(data_vec["edge_index"]))
        return Data(x=torch.tensor(data_vec["x"], dtype=torch.float),
                    edge_attr=torch.tensor(data_vec["edge_attr"], dtype=torch.float),
                    edge_index=torch.LongTensor(data_vec["edge_index"]))


class GNNEdgeWrapper(GNNWrapper):
    """
    Wrapper class for any graph NN with >1 convolutional layer and edge features,
    Inherits from GNNWrapper all the methods except for get_embeddings()
    Reason: GNN with edge attributes requires passing then as an additional argument in forward() method
    Args:
        model       :instance of pytorch geometric GNN model with node and edge attributes
        model_path  :path to trained model file
        device      :cpu or gpu
    """
    def __init__(self, model, model_path, name="", suffix="", device="cuda", noActDrop=False):
        super(GNNEdgeWrapper, self).__init__(model, model_path, name, suffix, device, noActDrop)

    def get_embeddings(self, torch_data):
        torch_data = torch_data.to(self.device) if self.device == "cuda" else torch_data
        if not self.reduced:
            convolution = F.dropout(self.activation(self.conv_layers[0](torch_data.x,
                                                                        torch_data.edge_index,
                                                                        torch_data.edge_attr)),
                                    p=self.dropp,
                                    training=False) 
        else:
            convolution = self.conv_layers[0](torch_data.x,
                                              torch_data.edge_index,
                                              torch_data.edge_attr) 
        for layer in self.conv_layers[1:-1]:
            if not self.reduced:
                convolution = F.dropout(self.activation(layer(convolution,
                                                            torch_data.edge_index,
                                                            torch_data.edge_attr)),
                                        p=self.dropp,
                                        training=False)
            else:
                convolution = layer(convolution,
                                    torch_data.edge_index,
                                    torch_data.edge_attr)
                
        convolution = self.conv_layers[-1](convolution,
                                           torch_data.edge_index,
                                           torch_data.edge_attr)
        if self.reduced:
            convolution = F.dropout(self.activation(convolution),
                                    p=self.dropp,
                                    training=False)
            
        return self.normalization(convolution)
    # def get_embeddings(self, torch_data):
    #     torch_data = torch_data.to(self.device) if self.device == "cuda" else torch_data
    #     convolution = self.conv_layers[0](torch_data.x,
    #                                         torch_data.edge_index,
    #                                         torch_data.edge_attr)
    #     for layer in self.conv_layers[1:-1]:
    #         convolution = layer(convolution,
    #                             torch_data.edge_index,
    #                             torch_data.edge_attr)

    #     convolution = self.conv_layers[-1](convolution,
    #                                        torch_data.edge_index,
    #                                        torch_data.edge_attr)
    #     convolution = F.dropout(self.activation(convolution),
    #                             p=0.3,
    #                             training=False)
        
    #     return self.normalization(convolution)


class GNNTrainer(GNNWrapper):
    """
    Trainer class to run training and testing of GNN models;
    Note: here model_path is a folder to save data
    """
    def __init__(self, model, model_path, name="", suffix="", device="cuda"):
        super(GNNTrainer, self).__init__(model, model_path, name, suffix, device)

    def train(self, dataloader, optimizer):
        self.model.train()
        for data in dataloader:
            dev_data = data.to(self.device) if self.device == "cuda" else data

            # Compute prediction error
            pred = self.model(dev_data)
            loss = self.loss_fn(pred, dev_data.y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def save_output(self, output):
        filepath = os.path.join(self.model_path, "accuracy_{}_{}.json".format(self.name, self.suffix))
        with open(filepath, "w") as f:
            f.write(json.dumps(output))

    def save_model(self):
        torch.save(self.model.state_dict(),
                   os.path.join(self.model_path, "{}_{}.model".format(self.name, self.suffix)))
