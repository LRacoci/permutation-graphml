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

import sys
sys.path.insert(0, '../datasets/darwin')
from graph_surface_dataset import *

from functions_helper import *

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

#@title Run training  { form-width: "30%" }
def train():
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
        theta
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

    tf.reset_default_graph()
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

    # How much time between logging and printing the current results.
    log_every_seconds = 20

    print("# (iteration number), T (elapsed seconds), "
        "Ltr (training loss), Lge (test/generalization loss), "
        "Ctr (training fraction nodes/edges labeled correctly), "
        "Str (training fraction examples solved correctly), "
        "Cge (test/generalization fraction nodes/edges labeled correctly), "
        "Sge (test/generalization fraction examples solved correctly)"
    )

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
        train_values = sess.run(
            {
                "step": step_op,
                "target": target_ph,
                "loss": loss_op_tr,
                "outputs": output_ops_tr
            }, 
            feed_dict=feed_dict
        )

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        if elapsed_since_last_log > log_every_seconds:
            last_log_time = the_time
            feed_dict, raw_graphs = create_feed_dict(
                    rand, batch_size_ge, num_nodes_min_max_ge, theta, input_ph, target_ph)
            test_values = sess.run({
                "target": target_ph,
                "loss": loss_op_ge,
                "outputs": output_ops_ge
            }, feed_dict=feed_dict)

            correct_tr, solved_tr = compute_accuracy(
                    train_values["target"], train_values["outputs"][-1], use_edges=True)
            correct_ge, solved_ge = compute_accuracy(
                    test_values["target"], test_values["outputs"][-1], use_edges=True)
            elapsed = time.time() - start_time
            losses_tr.append(train_values["loss"])
            corrects_tr.append(correct_tr)
            solveds_tr.append(solved_tr)
            losses_ge.append(test_values["loss"])
            corrects_ge.append(correct_ge)
            solveds_ge.append(solved_ge)
            logged_iterations.append(iteration)
            print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
                " {:.4f}, Cge {:.4f}, Sge {:.4f}".format(
                    iteration, elapsed, train_values["loss"], test_values["loss"],
                    correct_tr, solved_tr, correct_ge, solved_ge)
            )

    return raw_graphs, test_values

def visualize_results(raw_graphs, test_values):
    # This cell visualizes the results of training. You can visualize the
    # intermediate results by interrupting execution of the cell above, and running
    # this cell. You can then resume training by simply executing the above cell
    # again.

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
    max_graphs_to_plot = 6
    num_steps_to_plot = 4
    node_size = 120
    min_c = 0.3
    num_graphs = len(raw_graphs)
    targets = utils_np.graphs_tuple_to_data_dicts(test_values["target"])
    step_indices = np.floor(
            np.linspace(0, num_processing_steps_ge - 1,
                                num_steps_to_plot)).astype(int).tolist()
    outputs = list(
            zip(*(utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][i])
                for i in step_indices)))
    h = min(num_graphs, max_graphs_to_plot)
    w = num_steps_to_plot + 1
    fig = plt.figure(101, figsize=(18, h * 3))
    fig.clf()
    ncs = []
    for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
        if j >= h:
            break
        pos = get_node_dict(graph, "pos")
        ground_truth = target["nodes"][:, -1]
        # Ground truth.
        iax = j * (1 + num_steps_to_plot) + 1
        ax = fig.add_subplot(h, w, iax)
        plotter = GraphPlotter(ax, graph, pos)
        color = {}
        for i, n in enumerate(plotter.nodes):
            color[n] = np.array([1.0 - ground_truth[i], 0.0, ground_truth[i], 1.0
                                                    ]) * (1.0 - min_c) + min_c
        plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.set_facecolor([0.9] * 3 + [1.0])
        except AttributeError:
            ax.set_axis_bgcolor([0.9] * 3 + [1.0])
        ax.grid(None)
        ax.set_title("Ground truth\nSolution length: {}".format(
                plotter.solution_length))
        # Prediction.
        for k, outp in enumerate(output):
            iax = j * (1 + num_steps_to_plot) + 2 + k
            ax = fig.add_subplot(h, w, iax)
            plotter = GraphPlotter(ax, graph, pos)
            color = {}
            prob = softmax_prob_last_dim(outp["nodes"])
            for i, n in enumerate(plotter.nodes):
                color[n] = np.array([1.0 - prob[n], 0.0, prob[n], 1.0
                                                    ]) * (1.0 - min_c) + min_c
            plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(
                    step_indices[k] + 1, step_indices[-1] + 1))