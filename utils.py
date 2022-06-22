DEV_DIR = "/home/olgakononova/dev/"

import sys
sys.path.append(DEV_DIR)

import json
import os

from GNN_models.features import get_features_dim
from GNN_models.trf_models import TrfEdgeNet, TrfEdgeNetL5, TrfEdgeNetNoActDrop
from GNN_models.cgcnn_model import CGCNNet
from GNN_models.gnn_parent import GNNEdgeWrapper

model_map = {"TrfEdgeNet": TrfEdgeNet,
             "TrfEdgeNetL5": TrfEdgeNetL5,
             "TrfEdgeNetNoActDrop": TrfEdgeNetNoActDrop,
             "CGCNNet": CGCNNet   
            }

def get_atom_types_model(model_type, 
                         model_path, 
                         model_name, 
                         model_suffix, 
                         atom_types_file, 
                         device):
    atom_types_dict = json.loads(open(atom_types_file).read())

    num_node_features, num_edge_features = get_features_dim()
    num_classes = len(atom_types_dict)
    gnn_model = model_map[model_type]
    embed_model = GNNEdgeWrapper(gnn_model(num_node_features, num_edge_features, num_classes),
                                 model_path=os.path.join(model_path),
                                 name=model_name,
                                 suffix=model_suffix,
                                 device=device
                                )
    embed_model.load_model()
    embed_model.model.eval()
    return embed_model