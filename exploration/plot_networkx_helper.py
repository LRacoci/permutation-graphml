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

from functions_helper import *

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)


def visualize_example():
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


    min_nodes = 54 #@param {type:"slider", min:4, max:64, step:1}
    max_nodes = 57 #@param {type:"slider", min:4, max:64, step:1}

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


def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)

#@title Visualize results  { form-width: "30%" }

def visualize_results():
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
    step_indices = np.floor(np.linspace(0, num_processing_steps_ge - 1, num_steps_to_plot)).astype(int).tolist()
    outputs = list(zip(*(utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][i]) for i in step_indices)))
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
            color[n] = np.array([1.0 - ground_truth[i], 0.0, ground_truth[i], 1.0]) * (1.0 - min_c) + min_c
        plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
        ax.set_xticks([])
        ax.set_axis_on()
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
                color[n] = np.array([1.0 - prob[n], 0.0, prob[n], 1.0]) * (1.0 - min_c) + min_c
            plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
            ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(step_indices[k] + 1, step_indices[-1] + 1))