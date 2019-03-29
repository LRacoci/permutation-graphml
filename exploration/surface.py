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

#@title Helper functions  { form-width: "30%" }

# pylint: disable=redefined-outer-name


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
        nx.set_node_attributes(G = nxGraph, name ="pos", values = {n : val for n,val in enumerate(node_feature)})
        nx.set_edge_attributes(G = nxGraph, name ="features", values = 0)
        nxGraphs.append(nxGraph)
    
    return nxGraphs

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

    
def generate_raw_graphs(
	rand,
    num_examples = 2,
    min_max_nodes = 4,
):
	types=[
		'elliptic_paraboloid',
		'saddle',
		'torus',
		'ellipsoid',
		'elliptic_hyperboloid',
		'another'
	]
	min_nodes, max_nodes = min_max_nodes
	num_nodes = rand.randint(min_nodes, max_nodes)
	
	gen_graph = GenerateDataGraphSurface(type_dataset='elliptic_hyperboloid', num_surfaces=num_examples, num_points=num_nodes)
	epochs=1
	batch_size=10
	for epoch in range(epochs):
		print("\n########## epoch " + str(epoch+1) + " ##########")
		gen_trainig = gen_graph.train_generator( batch_size = batch_size )
		counter = 0
		for gt_graph, set_feature, in_graph in gen_trainig:
			print("---- batch ----")
			print("gt_graph: ", gt_graph)
			print("set_feature: ", set_feature)
			
			print("in_graph.shape: ", in_graph.shape)
			print("gt_graph.shape: ", gt_graph.shape)
			print("set_feature.shape: ", set_feature.shape)
			
			nxGraphs = darwin_batches_to_networkx_graphs(gt_graph, set_feature)
			return nxGraphs
			
			'''
			graphs_tuple = utils_np.networkxs_to_graphs_tuple(nxGraphs)
			print("graphs_tuple.shape : ", graphs_tuple.map(lambda a: a if a is None else a.shape, fields=graphs.ALL_FIELDS))
			saveable_string = graphs_tuple_dumps(graphs_tuple)
			with open("surf/surf{}.json".format(epoch+1), 'w') as file:
				file.write(saveable_string)
			break
			'''

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


num_examples = 2  #@param{type: 'integer'}


min_nodes = 4 #@param {type:"slider", min:4, max:64, step:1}
max_nodes = 5 #@param {type:"slider", min:4, max:64, step:1}

min_max_nodes = (min_nodes, max_nodes)

graphs = generate_raw_graphs(
    rand,
    num_examples,
    min_max_nodes,
)