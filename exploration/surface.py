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
	geo_density = None
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
    geo_density,
    source_ph,
    target_ph
):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
        min_max_nodes: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        geo_density: A `float` threshold parameters for the geographic threshold graph's
            threshold. Default= the number of nodes.
        source_ph: The source graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.

    Returns:
        feed_dict: The feed `dict` of source and target placeholders and data.
        raw_graphs: The `dict` of raw networkx graphs.
    """
    sources, targets, raw_graphs = generate_networkx_graphs(
        rand,
        batch_size,
        min_max_nodes,
    )
    source_graphs = utils_np.networkxs_to_graphs_tuple(sources)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {source_ph: source_graphs, target_ph: target_graphs}
    return feed_dict, raw_graphs

def compute_accuracy(target, output, use_nodes=False, use_edges=True):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
        target: A `graphs.GraphsTuple` that contains the target graph.
        output: A `graphs.GraphsTuple` that contains the output graph.
        use_nodes: A `bool` indicator of whether to compute node accuracy or not.
        use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved

#@title Run training  { form-width: "30%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.

# How much time between logging and printing the current results.
log_every_seconds = 20

print("# (iteration number), T (elapsed seconds), "
            "Ltr (training loss), Lge (test/generalization loss), "
            "Ctr (training fraction nodes/edges labeled correctly), "
            "Str (training fraction examples solved correctly), "
            "Cge (test/generalization fraction nodes/edges labeled correctly), "
            "Sge (test/generalization fraction examples solved correctly)")

start_time = time.time()
last_log_time = start_time
for iteration in range(last_iteration, num_training_iterations):
    last_iteration = iteration
    feed_dict, _ = create_feed_dict(
        rand,
        batch_size_tr,
        num_nodes_min_max_tr,
        theta,
        input_ph,
        target_ph
    )
    train_values = sess.run({
            "step": step_op,
            "target": target_ph,
            "loss": loss_op_tr,
            "outputs": output_ops_tr
    },
                                                    feed_dict=feed_dict)
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time
    if elapsed_since_last_log > log_every_seconds:
        last_log_time = the_time
        feed_dict, raw_graphs = create_feed_dict(
                rand, batch_size_ge, num_nodes_min_max_ge, theta, input_ph, target_ph)
        test_values = sess.run(
            {
                "target": target_ph,
                "loss": loss_op_ge,
                "outputs": output_ops_ge
            },
            feed_dict=feed_dict
        )
        correct_tr, solved_tr = compute_accuracy(
            train_values["target"],
            train_values["outputs"][-1],
            use_edges=True
        )
        correct_ge, solved_ge = compute_accuracy(
            test_values["target"],
            test_values["outputs"][-1],
            use_edges=True
        )
        elapsed = time.time() - start_time
        losses_tr.append(train_values["loss"])
        corrects_tr.append(correct_tr)
        solveds_tr.append(solved_tr)
        losses_ge.append(test_values["loss"])
        corrects_ge.append(correct_ge)
        solveds_ge.append(solved_ge)
        logged_iterations.append(iteration)
        print(
            "# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
            " {:.4f}, Cge {:.4f}, Sge {:.4f}".format(
                iteration, elapsed, train_values["loss"], test_values["loss"],
                correct_tr, solved_tr, correct_ge, solved_ge
            )
        )
#@title Visualize results  { form-width: "30%" }

# This cell visualizes the results of training. You can visualize the
# intermediate results by interrupting execution of the cell above, and running
# this cell. You can then resume training by simply executing the above cell
# again.

def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)


# Plot results curves.
fig = plt.figure(1, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)
# Loss.
y_tr = losses_tr
y_ge = losses_ge
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Loss across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss (binary cross-entropy)")
ax.legend()
# Correct.
y_tr = corrects_tr
y_ge = corrects_ge
ax = fig.add_subplot(1, 3, 2)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction correct across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction nodes/edges correct")
# Solved.
y_tr = solveds_tr
y_ge = solveds_ge
ax = fig.add_subplot(1, 3, 3)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction solved across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction examples solved")

# Plot graphs and results after each processing step.
# The white node is the start, and the black is the end. Other nodes are colored
# from red to purple to blue, where red means the model is confident the node is
# off the shortest path, blue means the model is confident the node is on the
# shortest path, and purplish colors mean the model isn't sure.
max_graphs_to_plot = 8 #@param{type:"slider", min:4, max:8, step:1}
num_steps_to_plot = 4 #@param{type:"slider", min:1, max:8, step:1}
node_size = 120 #@param{type:"slider", min:64, max:2048, step:1}
min_c = 0.3
num_graphs = len(raw_graphs)
targets = utils_np.graphs_tuple_to_data_dicts(test_values["target"])
step_indices = np.floor(
    np.linspace(
        0, num_processing_steps_ge - 1,
        num_steps_to_plot
    )
).astype(int).tolist()

outputs = list(zip(*(
    utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][i])
    for i in step_indices
)))
h = min(num_graphs, max_graphs_to_plot)
w = num_steps_to_plot + 1
fig = plt.figure(101, figsize=(18, h * 3))
fig.clf()
ncs = []
for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
    if j >= h:
        break
    ground_truth = target["nodes"][:, -1]
    # Ground truth.
    iax = j * (1 + num_steps_to_plot) + 1
    ax = fig.add_subplot(h, w, iax)
    plotter = GraphPlotter(ax, graph)
    color = {}
    for i, n in enumerate(plotter.nodes):
        color[n] = np.array(
            [1.0 - ground_truth[i], 0.0, ground_truth[i], 1.0]
        ) * (1.0 - min_c) + min_c
    plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        ax.set_facecolor([0.9] * 3 + [1.0])
    except AttributeError:
        ax.set_axis_bgcolor([0.9] * 3 + [1.0])
    ax.grid(None)
    ax.set_title(
        "Ground truth\nSolution length: {}"
            .format(plotter.solution_length)
    )
    # Prediction.
    for k, outp in enumerate(output):
        iax = j * (1 + num_steps_to_plot) + 2 + k
        ax = fig.add_subplot(h, w, iax)
        plotter = GraphPlotter(ax, graph)
        color = {}
        prob = softmax_prob_last_dim(outp["nodes"])
        for i, n in enumerate(plotter.nodes):
            color[n] = np.array(
                [1.0 - prob[n], 0.0, prob[n], 1.0]
            ) * (1.0 - min_c) + min_c
        plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
        ax.set_title(
            "Model-predicted\nStep {:02d} / {:02d}".format(
                step_indices[k] + 1,
                step_indices[-1] + 1
            )
        )