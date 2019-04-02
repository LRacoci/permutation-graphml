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


#@title Helper functions for setup training { form-width: "30%" }

def source_from_raw(raw):
    return raw.copy()

def target_from_raw(raw):
    return raw.copy()

def generate_networkx_graphs(rand, num_examples, min_max_nodes, geo_density):

    raw_graphs = generate_raw_graphs(rand, num_examples, min_max_nodes, geo_density)
    source_graphs = [source_from_raw(raw) for raw in raw_graphs]
    target_graphs = [target_from_raw(raw) for raw in raw_graphs]

    return source_graphs, target_graphs, raw_graphs


# pylint: disable=redefined-outer-name
def create_placeholders(rand, batch_size, min_max_nodes, geo_density):
    # Create some example data for inspecting the vector sizes.
    raw_graphs = generate_raw_graphs(rand, batch_size, min_max_nodes, geo_density)
    source_graphs = [source_from_raw(raw) for raw in raw_graphs]
    source_ph = utils_tf.placeholders_from_networkxs(
        source_graphs,
        force_dynamic_num_graphs=True
    )

    target_graphs = [target_from_raw(raw) for raw in raw_graphs]
    # print_graphs(target_graphs)

    target_ph = utils_tf.placeholders_from_networkxs(
        target_graphs,
        force_dynamic_num_graphs=True
    )
    return source_ph, target_ph


def create_loss_ops(target_op, output_ops):
    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
        tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]








#@title Set up model training and evaluation  { form-width: "30%" }

# The model we explore includes three components:
# - An "Encoder" graph net, which independently encodes the edge, node, and
#   global attributes (does not compute relations etc.).
# - A "Core" graph net, which performs N rounds of processing (message-passing)
#   steps. The input to the Core is the concatenation of the Encoder's output
#   and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
#   the processing step).
# - A "Decoder" graph net, which independently decodes the edge, node, and
#   global attributes (does not compute relations etc.), on each
#   message-passing step.
#
#                     Hidden(t)   Hidden(t+1)
#                        |            ^
#           *---------*  |  *------*  |  *---------*
#           |         |  |  |      |  |  |         |
# Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
#           |         |---->|      |     |         |
#           *---------*     *------*     *---------*
#
# The model is trained by supervised learning. Input graphs are procedurally
# generated, and output graphs have the same structure with the nodes and edges
# of the shortest path labeled (using 2-element 1-hot vectors). We could have
# predicted the shortest path only by labeling either the nodes or edges, and
# that does work, but we decided to predict both to demonstrate the flexibility
# of graph nets' outputs.
#
# The training loss is computed on the output of each processing step. The
# reason for this is to encourage the model to try to solve the problem in as
# few steps as possible. It also helps make the output of intermediate steps
# more interpretable.
#
# There's no need for a separate evaluate dataset because the inputs are
# never repeated, so the training loss is the measure of performance on graphs
# from the input distribution.
#
# We also evaluate how well the models generalize to graphs which are up to
# twice as large as those on which it was trained. The loss is computed only
# on the final processing step.
#
# Variables with the suffix _tr are training parameters, and variables with the
# suffix _ge are test/generalization parameters.
#
# After around 2000-5000 training iterations the model reaches near-perfect
# performance on graphs with between 8-16 nodes.

tf.reset_default_graph()

seed = 2 
rand = np.random.RandomState(seed=seed)

# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters.
num_training_iterations = 10000
theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
batch_size_tr = 32
batch_size_ge = 100
# Number of nodes per graph sampled uniformly from this range.
num_nodes_min_max_tr = (8, 17)
num_nodes_min_max_ge = (16, 33)

# Data.
# Input and target placeholders.
input_ph, target_ph = create_placeholders(
    rand,
    batch_size_tr,
    num_nodes_min_max_tr,
	None
)

# Connect the data to the model.
# Instantiate the model.
model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
# A list of outputs, one per processing step.
output_ops_tr = model(input_ph, num_processing_steps_tr)
output_ops_ge = model(input_ph, num_processing_steps_ge)

# Training loss.
loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
# Loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
# Test/generalization loss.
loss_ops_ge = create_loss_ops(target_ph, output_ops_ge)
loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

# Optimizer.
learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
step_op = optimizer.minimize(loss_op_tr)

# Lets an iterable of TF graphs be output from a session as NP graphs.
input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

#@title Reset session  { form-width: "30%" }

# This cell resets the Tensorflow session, but keeps the same computational
# graph.

try:
    sess.close()
except NameError:
    pass
sess = tf.Session()
sess.run(tf.global_variables_initializer())

last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []

#@title Helper functions for training { form-width: "30%" }

# pylint: disable=redefined-outer-name
def create_feed_dict(
    rand,
    batch_size,
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
