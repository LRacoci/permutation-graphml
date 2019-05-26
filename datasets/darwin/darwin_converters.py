#@title Converters to/from Darwin's format  { form-width: "20%" }
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

def darwin_batches_to_networkx_graphs(adjs_gt, node_features, adjs_inp, set_segmentation):
    '''
    Args:
        adjs_gt : adjacency matrix of ground truth, shape = (num_graphs, num_nodes, num_nodes)
        node_features :  Matrix of node features, shape = (num_graphs, num_nodes, num_node_features)
        adjs_inp : adjacency matrix of input graph, shape = (num_graphs, num_nodes, num_nodes)

    Returns:
        graphs_tuple : GraphTuple from graph_nets
    '''
    nxGraphs = []
    for adj_gt, node_feature, adj_inp, set_segm in zip(adjs_gt, node_features, adjs_inp,set_segmentation):
        nxGraph = nx.from_numpy_matrix(adj_inp, create_using=nx.DiGraph)
        nx.set_node_attributes(G = nxGraph, name ="rgbxy", values = {
            n : val 
            for n,val in enumerate(node_feature)
        })
        print(set_segm)
        print(np.uint32(node_feature[:,-2:]))
        nx.set_node_attributes(G = nxGraph, name ="resp", values = {
            n : val
            for n,val in enumerate(set_segm[np.uint32(node_feature[:,-2:])])
        })
        nx.set_edge_attributes(G = nxGraph, name ="resp", values = {
            (u,v) : adj_gt[u][v]
            for (u,v) in nxGraph.edges
        })
        nxGraphs.append(nxGraph)

    return nxGraphs

import json
def json_default(obj) :
    class_name = obj.__class__.__name__
    serialization = {
        'int64' : int,
        'int32' : int,
        'ndarray' : list
    }
    if class_name in serialization:
        return serialization[class_name](obj)
    else:
        print("Unserializable object {} of type {}".format(obj, type(obj)))
        raise TypeError(
            "Unserializable object {} of type {}".format(obj, class_name)
        )

def json_dumps(obj, indent = 4, default = json_default) :
    return json.dumps(obj, indent = indent, default = default)

def graphs_tuple_dumps(graphs_tuple):
    data_dicts = utils_np.graphs_tuple_to_data_dicts(graphs_tuple)
    return json_dumps(data_dicts)

def graphs_tuple_loads(string_dump):
    data_dicts = json.loads(string_dump)
    for data_dict in data_dicts:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dicts)
    return graphs_tuple


def graphs_tuples_to_darwin_batches(graph_nets):
    '''
    Args:
        graphs_tuple : GraphTuple from graph_nets
    Returns:
        graph : adjacency matrix of the graph, shape = (num_graphs, num_nodes, num_nodes)
        node_features :  Matrix of node features, shape = (num_graphs, num_nodes, num_node_features)
    '''
    adjs_gt = []
    node_features = []
    adjs_inp = []
    
    data_dicts = utils_np.graphs_tuple_to_data_dicts(graph_nets)
    for data_dict in data_dicts:
        nodes = data_dict['nodes']
        num_nodes= len(nodes)
        
        adj_inp = np.zeros(shape = (num_nodes, num_nodes))
        adj_gt = np.zeros(shape = (num_nodes, num_nodes))
        
        senders = data_dict['senders']
        receivers = data_dict['receivers']
        edges = data_dict['edges']
        
        adj_inp[senders, receivers] = 1.0
        adjs_gt[senders, receivers] = edges
        
        adjs_inp.append(adj_inp)
        adjs_gt.append(adj_gt)
        node_features.append(nodes)
    
    return np.array(adjs_gt), np.array(node_features), np.array(adjs_inp)

