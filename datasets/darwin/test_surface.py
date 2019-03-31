from graph_surface_dataset import *
from darwin_converters import *

np.set_printoptions(threshold=np.nan)

types = [
    'elliptic_paraboloid',
    'saddle',
    'torus',
    'ellipsoid',
    'elliptic_hyperboloid',
    'another'
]
gen_graph = GenerateDataGraphSurface(type_dataset='elliptic_hyperboloid', num_surfaces=2, num_points=4)
print(gen_graph.target)
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
        graphs_tuple = utils_np.networkxs_to_graphs_tuple(nxGraphs)
        print("graphs_tuple.shape : ", graphs_tuple.map(lambda a: a if a is None else a.shape, fields=graphs.ALL_FIELDS))
        saveable_string = graphs_tuple_dumps(graphs_tuple)

        #Check:
        graphs_tuple_test = graphs_tuple_loads(saveable_string)
        test_graph, test_feature = graphs_tuples_to_darwin_batches(graphs_tuple_test)
        check_condition = np.all(test_graph == gt_graph) and np.all(set_feature == test_feature)
        if check_condition:
            print("\n\n---------------------------------\nAll functions working")
        
        with open("surf/surf{}.json".format(epoch+1), 'w') as file:
            file.write(saveable_string)