from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '../datasets/darwin')
from graph_surface_dataset import *

#@title Imports  { form-width: "30%" }


import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import tensorflow as tf

from plot_networkx_helper import *

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)


def print_graphs_tuple(graphs_tuple):
	print("Shapes of `GraphsTuple`'s fields:")
	print(graphs_tuple.map(lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS))
	print("\nData contained in `GraphsTuple`'s fields:")
	print("globals:\n{}".format(graphs_tuple.globals))
	print("nodes:\n{}".format(graphs_tuple.nodes))
	print("edges:\n{}".format(graphs_tuple.edges))
	print("senders:\n{}".format(graphs_tuple.senders))
	print("receivers:\n{}".format(graphs_tuple.receivers))
	print("n_node:\n{}".format(graphs_tuple.n_node))
	print("n_edge:\n{}".format(graphs_tuple.n_edge))


SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#@title Helper functions  { form-width: "20%" }

# pylint: disable=redefined-outer-name


def print_graph(g):
    for s, t, w in g.edges(data=True):
        if 'features' not in w:
            print(s, t, w, "(problem)")
        else:
            print(s, t, w, "(ok)")

def print_graphs(gs):
    for g in gs:
        print("----------------------------------------------------------------------------")
        print_graph(g)
        print("----------------------------------------------------------------------------")

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))

def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot

def create_feature(feature, fields):
    return np.hstack([np.array(feature[field], dtype=float) for field in fields])
    
def generate_raw_graphs(rand, batch_size, min_max_nodes, geo_density):  
    num_nodes = rand.random_integers(*min_max_nodes)
    surface_type = str(rand.choice(SURFACE_TYPES, 1)[0])
    gen_graph = GenerateDataGraphSurface(type_dataset=surface_type, num_surfaces=batch_size, num_points=num_nodes)
    epochs=1
    for epoch in range(epochs):
        gen_trainig = gen_graph.train_generator( batch_size = batch_size )
        counter = 0
        for gt_graph, set_feature, in_graph in gen_trainig:
            print ("gt_graph.shape = ", gt_graph.shape)
            print ("set_feature.shape = ", set_feature.shape)
            print ("surface_type = ", surface_type)
            nxGraphs = darwin_batches_to_networkx_graphs(gt_graph, set_feature, surface_type)
            return nxGraphs

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

def darwin_batches_to_networkx_graphs(graphs, node_features, surface_type):
    '''
    Args:
        graph : adjacency matrix of the graph, shape = (num_graphs, num_nodes, num_nodes)
        node_features :  Matrix of node features, shape = (num_graphs, num_nodes, num_node_features)

    Returns:
        graphs_tuple : GraphTuple from graph_nets
    '''
    global SURFACE_TYPES
    nxGraphs = []
    for graph, node_feature in zip(graphs, node_features):
        nxGraph = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
        nx.set_node_attributes(G = nxGraph, name ="pos", values = {n : val for n,val in enumerate(node_feature)})
        nx.set_edge_attributes(G = nxGraph, name ="distance", values = {
            (u,v) : np.linalg.norm(node_feature[u] - node_feature[v])
            for (u,v) in nxGraph.edges
        })
        nxGraph.graph['type'] = to_one_hot(SURFACE_TYPES.index(surface_type), len(SURFACE_TYPES))
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

#@title #Visualize example graphs  { form-width: "30%" }
def rgb_from_hex(h):
    if h[0] != '#' or len(h) != 7:
        raise ValueError("'{}' should be '#' followed by 6 HEX chars".format(h))
    h = h.lstrip('#')
    h = h.upper()
    invalid_chars = set(list(h)) - set(list("0123456789ABCDEF"))
    if invalid_chars:
        raise ValueError("Incorrect chars: \{{}\}".format(', '.format(invalid_chars)))
    return tuple(int(h[i:i+2], 16)/255.0 for i in [0, 2 ,4])

#@markdown ##General Vizual Params

node_size=200 #@param{type:"slider", min:128, max:2048, step:1}
node_hex_color = "#808080" #@param {type:"string"}
node_color = rgb_from_hex(node_hex_color)
node_linewidth=1.0 #@param{type:"slider", min:0.1, max:3.0, step:0.1}
edge_width=0.2 #@param{type:"slider", min:0.1, max:3.0, step:0.1}
edge_style = "dashed" #@param ["solid", "dashed", "dotted", "dashdot"]
start_color="w"
end_color="k"

#@markdown ##Solution Vizual Paramters
solution_node_hex_color = "#3DFF3D" #@param {type:"string"}
solution_node_color = rgb_from_hex(solution_node_hex_color)
solution_node_linewidth=0.6 #@param{type:"slider", min:0.1, max:6.0, step:0.1}
solution_edge_width=6.0 #@param{type:"slider", min:0.1, max:4.0, step=0.1}
solution_edge_style = "solid" #@param ["solid", "dashed", "dotted", "dashdot"]


#@markdown ##Specific Parameters

seed = 5  #@param{type: 'integer'}
rand = np.random.RandomState(seed=seed)

num_examples = 4  #@param{type: 'integer'}


min_nodes = 34 #@param {type:"slider", min:4, max:64, step:1}
max_nodes = 36 #@param {type:"slider", min:4, max:64, step:1}

theta = 12  #@param{type:"slider", min:4, max:64, step:1}
#@markdown Large values (1000+) make trees. Try 20-60 for good non-trees.

horizontal_length = 20 #@param{type: 'integer'}
graphs_per_column = 2 #@param{type: 'integer'}

min_max_nodes = (min_nodes, max_nodes)

graphs = generate_raw_graphs(
    rand,
    num_examples,
    min_max_nodes,
    theta
)

num = min(num_examples, 16)
size = horizontal_length/graphs_per_column
w = graphs_per_column
h = int(np.ceil(num / w))
fig = plt.figure(40, figsize=(w * size, h * size))
fig.clf()
for j, graph in enumerate(graphs):
    ax = fig.add_subplot(h, w, j + 1, projection='3d')
    points_coord_dict = nx.get_node_attributes(graph,'pos')
    points_coord = []
    for u in points_coord_dict:
        points_coord.append(points_coord_dict[len(points_coord)])
    points_coord = np.array(points_coord)                   
    print(points_coord)
    x = points_coord[:,0]
    y = points_coord[:,1]
    z = points_coord[:,2]

    list_edges = []
    #plot lines from edges
    for u,v in graph.edges:
        line = plt3d.art3d.Line3D(
            [x[u],x[v]], 
            [y[u],y[v]], 
            [z[u],z[v]], 
            linewidth=0.4, 
            c="black", 
            alpha=1.
        )
        ax.add_line(line)

    ax.scatter(x,y,z, marker='.', s=15, c="blue", alpha=0.6)
