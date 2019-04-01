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

def darwin_batches_to_networkx_graphs(graphs, node_features):
    '''
    Args:
        graph : adjacency matrix of the graph, shape = (num_graphs, num_nodes, num_nodes)
        node_features :  Matrix of node features, shape = (num_graphs, num_nodes, num_node_features)

    Returns:
        graphs_tuple : GraphTuple from graph_nets
    '''
    nxGraphs = []
    for graph, node_feature in zip(graphs, node_features):
        nxGraph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
        nx.set_node_attributes(G = nxGraph, name ="features", values = {n : val for n,val in enumerate(node_feature)})
        nx.set_edge_attributes(G = nxGraph, name ="features", values = 0)
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
    adjs = []
    node_features = []
    data_dicts = utils_np.graphs_tuple_to_data_dicts(graph_nets)
    for data_dict in data_dicts:
        nodes = data_dict['nodes']
        num_nodes= len(nodes)
        adj = np.zeros(shape = (num_nodes, num_nodes))
        senders = data_dict['senders']
        receivers = data_dict['receivers']
        adj[senders, receivers] = 1
        adjs.append(adj)
        node_features.append(nodes)

    return np.array(adjs), np.array(node_features)

