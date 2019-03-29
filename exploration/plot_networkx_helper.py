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

#@title Graph Plot Helper Class  { form-width: "30%" }

def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}

# pylint: disable=redefined-outer-name
class GraphPlotter(object):

    def __init__(self, ax, graph):
        self._ax = ax
        self._graph = graph
        self._pos = get_node_dict(graph, "pos")
        self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
        self._solution_length = None
        self._nodes = None
        self._edges = None
        self._start_nodes = None
        self._end_nodes = None
        self._solution_nodes = None
        self._intermediate_solution_nodes = None
        self._solution_edges = None
        self._non_solution_nodes = None
        self._non_solution_edges = None
        self._ax.set_axis_off()

    @property
    def solution_length(self):
        if self._solution_length is None:
            self._solution_length = len(self._solution_edges)
        return self._solution_length

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._graph.nodes()
        return self._nodes

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._graph.edges()
        return self._edges

    @property
    def start_nodes(self):
        if self._start_nodes is None:
            self._start_nodes = [
                n for n in self.nodes
                if self._graph.node[n].get("start", False)
            ]
        return self._start_nodes

    @property
    def end_nodes(self):
        if self._end_nodes is None:
            self._end_nodes = [
                n for n in self.nodes
                if self._graph.node[n].get("end", False)
            ]
        return self._end_nodes

    @property
    def solution_nodes(self):
        if self._solution_nodes is None:
            self._solution_nodes = [
                n for n in self.nodes
                if self._graph.node[n].get("solution", False)
            ]
        return self._solution_nodes

    @property
    def intermediate_solution_nodes(self):
        if self._intermediate_solution_nodes is None:
            self._intermediate_solution_nodes = [
                    n for n in self.nodes
                    if self._graph.node[n].get("solution", False) and
                    not self._graph.node[n].get("start", False) and
                    not self._graph.node[n].get("end", False)
            ]
        return self._intermediate_solution_nodes

    @property
    def solution_edges(self):
        if self._solution_edges is None:
            self._solution_edges = [
                    e for e in self.edges
                    if self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._solution_edges

    @property
    def non_solution_nodes(self):
        if self._non_solution_nodes is None:
            self._non_solution_nodes = [
                    n for n in self.nodes
                    if not self._graph.node[n].get("solution", False)
            ]
        return self._non_solution_nodes

    @property
    def non_solution_edges(self):
        if self._non_solution_edges is None:
            self._non_solution_edges = [
                e for e in self.edges
                if not self._graph.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._non_solution_edges

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if type(collection) is list:
            # This is for compatibility with newer matplotlib.
            collection = collection[0]

        if collection is not None and zorder is not None:
            collection.set_zorder(zorder)
        return collection

    def draw_nodes(self, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        if (
            "node_color" in kwargs and
            isinstance(kwargs["node_color"], collections.Sequence) and
            len(kwargs["node_color"]) in {3, 4} and
            not isinstance(
                kwargs["node_color"][0],
                (collections.Sequence, np.ndarray)
            )
        ):
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(
                np.array(kwargs["node_color"])[None],
                [num_nodes, 1]
            )
        return self._draw(nx.draw_networkx_nodes, **kwargs)

    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(nx.draw_networkx_edges, **kwargs)

    def draw_graph(
        self,
        node_size=200,
        node_color=(0.4, 0.8, 0.4),
        node_linewidth=1.0,
        edge_width=1.0
    ):
        # Plot nodes.
        self.draw_nodes(
            nodelist=self.nodes,
            node_size=node_size,
            node_color=node_color,
            linewidths=node_linewidth,
            zorder=20
        )
        # Plot edges.
        self.draw_edges(
            edgelist=self.edges,
            width=edge_width,
            zorder=10
        )

    def draw_graph_with_solution(
        self,
        node_size=200,
        node_color=(0.5, 0.7, 0.5),
        node_linewidth=1.0,
        start_color="w",
        end_color="k",
        edge_width=1.0,
        edge_style = "dashed",
        solution_node_color = (0.2, 1.0, 0.2),
        solution_node_linewidth=3.0,
        solution_edge_width=3.0,
        solution_edge_style = "solid"
    ):
        node_border_color = (0.0, 0.0, 0.0, 1.0)
        node_collections = {}
        # Plot start nodes.
        node_collections["start nodes"] = self.draw_nodes(
            nodelist=self.start_nodes,
            node_size=node_size,
            node_color=start_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=100
        )
        # Plot end nodes.
        node_collections["end nodes"] = self.draw_nodes(
            nodelist=self.end_nodes,
            node_size=node_size,
            node_color=end_color,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=90
        )
        # Plot intermediate solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.intermediate_solution_nodes]
        else:
            c = solution_node_color
        node_collections["intermediate solution nodes"] = self.draw_nodes(
            nodelist=self.intermediate_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=solution_node_linewidth,
            edgecolors=node_border_color,
            zorder=80
        )
        # Plot solution edges.
        node_collections["solution edges"] = self.draw_edges(
            edgelist=self.solution_edges,
            width=solution_edge_width,
            style=solution_edge_style,
            zorder=70
        )
        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
        node_collections["non-solution nodes"] = self.draw_nodes(
            nodelist=self.non_solution_nodes,
            node_size=node_size,
            node_color=c,
            linewidths=node_linewidth,
            edgecolors=node_border_color,
            zorder=20
        )
        # Plot non-solution edges.
        node_collections["non-solution edges"] = self.draw_edges(
            edgelist=self.non_solution_edges,
            width=edge_width,
            style=edge_style,
            zorder=10
        )
        # Set title as solution length.
        self._ax.set_title("Solution length: {}".format(self.solution_length))
        return node_collections


# pylint: enable=redefined-outer-name