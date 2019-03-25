import sonnet as snt

class GraphIndependent(snt.AbstractModule):
  """A graph block that applies models to the graph elements independently.
  The inputs and outputs are graphs. The corresponding models are applied to
  each element of the graph (edges, nodes and globals) in parallel and
  independently of the other elements. It can be used to encode or
  decode the elements of a graph.
  """

  def __init__(self,
               edge_model_fn=None,
               node_model_fn=None,
               global_model_fn=None,
               name="graph_independent"):
    """Initializes the GraphIndependent module.
    Args:
      edge_model_fn: A callable that returns an edge model function. The
        callable must return a Sonnet module (or equivalent). If passed `None`,
        will pass through inputs (the default).
      node_model_fn: A callable that returns a node model function. The callable
        must return a Sonnet module (or equivalent). If passed `None`, will pass
        through inputs (the default).
      global_model_fn: A callable that returns a global model function. The
        callable must return a Sonnet module (or equivalent). If passed `None`,
        will pass through inputs (the default).
      name: The module name.
    """
    super(GraphIndependent, self).__init__(name=name)

    with self._enter_variable_scope():
      # The use of snt.Module below is to ensure the ops and variables that
      # result from the edge/node/global_model_fns are scoped analogous to how
      # the Edge/Node/GlobalBlock classes do.
      if edge_model_fn is None:
        self._edge_model = lambda x: x
      else:
        self._edge_model = snt.Module(
            lambda x: edge_model_fn()(x), name="edge_model")  # pylint: disable=unnecessary-lambda
      if node_model_fn is None:
        self._node_model = lambda x: x
      else:
        self._node_model = snt.Module(
            lambda x: node_model_fn()(x), name="node_model")  # pylint: disable=unnecessary-lambda
      if global_model_fn is None:
        self._global_model = lambda x: x
      else:
        self._global_model = snt.Module(
            lambda x: global_model_fn()(x), name="global_model")  # pylint: disable=unnecessary-lambda

  def _build(self, graph):
    """Connects the GraphIndependent.
    Args:
      graph: A `graphs.GraphsTuple` containing non-`None` edges, nodes and
        globals.
    Returns:
      An output `graphs.GraphsTuple` with updated edges, nodes and globals.
    """
    return graph.replace(
        edges=self._edge_model(graph.edges),
        nodes=self._node_model(graph.nodes),
        globals=self._global_model(graph.globals))
