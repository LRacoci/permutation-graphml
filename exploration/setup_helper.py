#@title Helper functions for setup training { form-width: "30%" }

def source_from_raw(raw):
    source = nx.DiGraph()
    # Nodes
    fields = ('pos',)
    for node, feature in raw.nodes(data=True):
        #print(feature)
        source.add_node(
            node, features=create_feature(feature, fields)
        )
    # Edges
    fields = ('distance',)
    for receiver, sender, feature in raw.edges(data=True):
        source.add_edge(
            sender, receiver, features=create_feature(feature, fields)
        )

    source.graph["features"] = raw.graph["type"] * 0

    return source

def target_from_raw(raw):
    target = nx.DiGraph()
    # Nodes
    target.add_node(0, features=np.array([0.0]))
    target.add_node(1, features=np.array([1.0]))
    # Edges
    target.add_edge(0, 1, features=np.array([0.0]))
    target.graph['features'] = raw.graph['type']
    return target

def generate_networkx_graphs(rand, num_examples, min_max_nodes, geo_density):
    """Generate graphs for training.

    Args:
        rand: A random seed (np.RandomState instance).
        num_examples: Total number of graphs to generate.
        min_max_nodes: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        geo_density: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Default= the number of nodes.

    Returns:
        source_graphs: The list of source graphs.
        target_graphs: The list of output graphs.
        raw_graphs: The list of generated graphs.
    """
    raw_graphs = generate_raw_graphs(rand, num_examples, min_max_nodes, geo_density)
    source_graphs = [source_from_raw(raw) for raw in raw_graphs]
    target_graphs = [target_from_raw(raw) for raw in raw_graphs]

    return source_graphs, target_graphs, raw_graphs


# pylint: disable=redefined-outer-name
def create_placeholders(rand, batch_size, min_max_nodes, geo_density):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
        min_max_nodes: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        geo_density: A `float` threshold parameters for the geographic threshold graph's
            threshold. Default= the number of nodes.

    Returns:
        source_ph: The source graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.
    """
    # Create some example data for inspecting the vector sizes.
    raw_graphs = generate_raw_graphs(
        rand,
        batch_size,
        min_max_nodes,
        geo_density=geo_density
    )
    # Source
    source_graphs = [source_from_raw(raw) for raw in raw_graphs]
    source_ph = utils_tf.placeholders_from_networkxs(
        source_graphs,
        force_dynamic_num_graphs=True
    )
    # Target
    target_graphs = [target_from_raw(raw) for raw in raw_graphs]
    target_ph = utils_tf.placeholders_from_networkxs(
        target_graphs,
        force_dynamic_num_graphs=True
    )
    
    return source_ph, target_ph


def create_loss_ops(target_op, output_ops):
    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.globals, output_op.globals)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]