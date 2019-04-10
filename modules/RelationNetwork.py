from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
import sonnet as snt
import tensorflow as tf

class RelationNetwork(snt.AbstractModule):

  """Implementation of a Relation Network.
  See https://arxiv.org/abs/1706.01427 for more details.
  The global and edges features of the input graph are not used, and are
  allowed to be `None` (the receivers and senders properties must be present).
  The output graph has updated, non-`None`, globals.
  """

  def __init__(self,
               edge_model_fn,
               global_model_fn,
               reducer=tf.unsorted_segment_sum,
               name="relation_network"):
    """Initializes the RelationNetwork module.
    Args:
      edge_model_fn: A callable that will be passed to EdgeBlock to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see EdgeBlock for details).
      global_model_fn: A callable that will be passed to GlobalBlock to perform
        per-global computations. The callable must return a Sonnet module (or
        equivalent; see GlobalBlock for details).
      reducer: Reducer to be used by GlobalBlock to aggregate edges. Defaults to
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(RelationNetwork, self).__init__(name=name)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn,
          use_edges=False,
          use_receiver_nodes=True,
          use_sender_nodes=True,
          use_globals=False)

      self._global_block = blocks.GlobalBlock(
          global_model_fn=global_model_fn,
          use_edges=True,
          use_nodes=False,
          use_globals=False,
          edges_reducer=reducer)

  def _build(self, graph):
    """Connects the RelationNetwork.
    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s, except for the edges
        and global properties which may be `None`.
    Returns:
      A `graphs.GraphsTuple` with updated globals.
    Raises:
      ValueError: If any of `graph.nodes`, `graph.receivers` or `graph.senders`
        is `None`.
    """
    output_graph = self._global_block(self._edge_block(graph))
    return graph.replace(globals=output_graph.globals)