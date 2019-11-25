from models.components.netFrame import *

REGISTRY = {}
REGISTRY["mlp"] = Net_MLP
REGISTRY["cnn2mlp"] = net_frame_cnn_to_mlp