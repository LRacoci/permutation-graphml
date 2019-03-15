import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from sklearn.decomposition import PCA
from sklearn import preprocessing
import community
import pickle
import re

#-------import data
epsilon = 1e-12
#import time
#np.random.seed(seed=int(time.time()))

#-------
'''
def citeseer_ego():
    _, _, G = data.Graph_load(dataset='citeseer')
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    return graphs
'''
#-------

def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G

def perturb(graph_list, p_del, p_add=None):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        trials = np.random.binomial(1, p_del, size=G.number_of_edges())
        edges = list(G.edges())
        i = 0
        for (u, v) in edges:
            if trials[i] == 1:
                G.remove_edge(u, v)
            i += 1
        if p_add is None:
            num_nodes = G.number_of_nodes()
            p_add_est = np.sum(trials) / (num_nodes * (num_nodes - 1) / 2 -
                    G.number_of_edges())
        else:
            p_add_est = p_add

        nodes = list(G.nodes())
        tmp = 0
        for i in range(len(nodes)):
            u = nodes[i]
            trials = np.random.binomial(1, p_add_est, size=G.number_of_nodes())
            j = 0
            for j in range(i+1, len(nodes)):
                v = nodes[j]
                if trials[j] == 1:
                    tmp += 1
                    G.add_edge(u, v)
                j += 1

        perturbed_graph_list.append(G)
    return perturbed_graph_list



def perturb_new(graph_list, p):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_remove_count = 0
        for (u, v) in list(G.edges()):
            if np.random.rand()<p:
                G.remove_edge(u, v)
                edge_remove_count += 1
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u,v)) and (u!=v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)


def save_prediction_histogram(y_pred_data, fname_pred, max_num_node, bin_n=20):
    bin_edge = np.linspace(1e-6, 1, bin_n + 1)
    output_pred = np.zeros((bin_n, max_num_node))
    for i in range(max_num_node):
        output_pred[:, i], _ = np.histogram(y_pred_data[:, i, :], bins=bin_edge, density=False)
        # normalize
        output_pred[:, i] /= np.sum(output_pred[:, i])
    imsave(fname=fname_pred, arr=output_pred, origin='upper', cmap='Greys_r', vmin=0.0, vmax=3.0 / bin_n)


# directly get graph statistics from adj, obsoleted
def decode_graph(adj, prefix):
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    # G.remove_nodes_from(nx.isolates(G))
    print('num of nodes: {}'.format(G.number_of_nodes()))
    print('num of edges: {}'.format(G.number_of_edges()))
    G_deg = nx.degree_histogram(G)
    G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
    print('average degree: {}'.format(sum(G_deg_sum) / G.number_of_nodes()))
    if nx.is_connected(G):
        print('average path length: {}'.format(nx.average_shortest_path_length(G)))
        print('average diameter: {}'.format(nx.diameter(G)))
    G_cluster = sorted(list(nx.clustering(G).values()))
    print('average clustering coefficient: {}'.format(sum(G_cluster) / len(G_cluster)))
    cycle_len = []
    cycle_all = nx.cycle_basis(G, 0)
    for item in cycle_all:
        cycle_len.append(len(item))
    print('cycles', cycle_len)
    print('cycle count', len(cycle_len))
    draw_graph(G, prefix=prefix)


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G

# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


# pick the first connected component
def pick_connected_component(G):
    node_list = nx.node_connected_component(G,0)
    return G.subgraph(node_list)

def pick_connected_component_new(G):
    num_nodes = G.number_of_nodes()
    adj_list = [[] for x in range(num_nodes)]
    for edge_ in G.edges():
      adj_list[edge_[0]].append(edge_[1])
      adj_list[edge_[1]].append(edge_[0])
    for id,adj in enumerate(adj_list):
        id_min = min(adj)
        if id<id_min and id>=1:
        # if id<id_min and id>=4:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G

import os.path
# load a list of graphs
def load_graph_list(fname,is_real=True):
    print("Exist " + fname + " :", os.path.exists(fname))
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    for i in range(len(graph_list)):
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list


def export_graphs_to_txt(g_list, output_filename_prefix):
    i = 0
    for G in g_list:
        f = open(output_filename_prefix + '_' + str(i) + '.txt', 'w+')
        for (u, v) in G.edges():
            idx_u = G.nodes().index(u)
            idx_v = G.nodes().index(v)
            f.write(str(idx_u) + '\t' + str(idx_v) + '\n')
        i += 1

def snap_txt_output_to_nx(in_fname):
    G = nx.Graph()
    with open(in_fname, 'r') as f:
        for line in f:
            if not line[0] == '#':
                splitted = re.split('[ \t]', line)

                # self loop might be generated, but should be removed
                u = int(splitted[0])
                v = int(splitted[1])
                if not u == v:
                    G.add_edge(int(u), int(v))
    return G

def test_perturbed():

    graphs = []
    for i in range(100,101):
        for j in range(4,5):
            for k in range(500):
                graphs.append(nx.barabasi_albert_graph(i,j))
    g_perturbed = perturb(graphs, 0.9)
    print([g.number_of_edges() for g in graphs])
    print([g.number_of_edges() for g in g_perturbed])

'''
if __name__ == '__main__':
    #test_perturbed()
    #graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_community4_4_128_train_0.dat')
    #graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_community4_4_128_pred_2500_1.dat')
    graphs = load_graph_list('eval_results/mmsb/' + 'community41.dat')

    for i in range(0, 160, 16):
        draw_graph_list(graphs[i:i+16], 4, 4, fname='figures/community4_' + str(i))
'''


# ----------------------------------------------------------------------------------------
# Author: Darwin Saire Pilco
# ----------------------------------------------------------------------------------------

def generate_feature_list(G_list, num_perm, layout='spring', is_single=False, k=1):

  def noempty( var_dict ):
    return ((var_dict and True) or False)

  dim = 2
  G_list_relabel = []
  possition = []
  pos_ = []
  for i,G in enumerate(G_list):
    if layout=='spring':
        pos = nx.spring_layout(G,dim=dim,k=k/np.sqrt(G.number_of_nodes()),iterations=100)
        # pos = nx.spring_layout(G)
    elif layout=='spectral':
        pos = nx.spectral_layout(G,dim=dim)

    pos_less_dim = {}
    dict_relabel = {}
    arr = np.zeros((G.number_of_nodes(), dim))
    for j, (key, value) in enumerate(pos.items()):
        arr[j,:] = value[0:2] #value_standardizing
        pos_less_dim[j] = value[0:2]
        dict_relabel[key] = j
    possition.append(arr)
    pos_.append(pos_less_dim)
    G_ = nx.relabel_nodes(G,dict_relabel)
    G_list_relabel.append(G_)

  possition_perm = []
  pos_perm = []
  G_list_relabel_perm = []

  for k_ in range(len(possition)):
    perm_set = []#set()

    while len(perm_set) < num_perm:
      index_perm = np.random.permutation(G_list_relabel[k_].number_of_nodes())
      is_repeat = False
      for m in range(len(perm_set)):
        tmp_repeat = np.array_equal(perm_set[m], index_perm)
        is_repeat = is_repeat and tmp_repeat
      if is_repeat == False:
        perm_set.append(index_perm)


    for l in range(num_perm):
      index_perm = perm_set[l]
      adjmtrx = np.array(nx.to_numpy_matrix(G_list_relabel[k_]))
      adjmtrx_perm = np.zeros((adjmtrx.shape[0],adjmtrx.shape[1]))
      for u_ in range(adjmtrx.shape[0]):
        for v_ in range(adjmtrx.shape[1]):
          adjmtrx_perm[index_perm[u_]][index_perm[v_]] = adjmtrx[u_][v_]
          adjmtrx_perm[index_perm[v_]][index_perm[u_]] = adjmtrx[v_][u_]
      G_perm = nx.to_networkx_graph(data=adjmtrx_perm)
      G_list_relabel_perm.append(G_perm)

      pos_index_perm = {}
      pos_perm_t = np.zeros((possition[k_].shape[0],possition[k_].shape[1]))
      for l in range(G_list_relabel[k_].number_of_nodes()):
        pos_index_perm[index_perm[l]] = pos_[k_][l]
        pos_perm_t[index_perm[l]] = possition[k_][l]
      possition_perm.append(pos_perm_t)
      pos_perm.append(pos_index_perm)

  return possition_perm, pos_perm, G_list_relabel_perm

def shuffle_list(*ls):
  l =list(zip(*ls))
  shuffle(l)
  #return zip(*l)
  return map(list, zip(*l))

def draw_graph(G_arr, row, col, pos, fname = 'Results/figures'):
    '''draw a numpy of graphs [num_graph,num_nodes,num_nodes], with coordinates'''
    plt.switch_backend('agg')
    for j in range(0,G_arr.shape[0],row+col):
        plt.figure()
        right = min(j+row+col,G_arr.shape[0])
        for i, id in enumerate(range(j,right)):
          plt.subplot(row,col,i+1)
          plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
          plt.axis("off")

          G = nx.to_networkx_graph(data=G_arr[id])

          nx.draw_networkx_nodes(G, pos[id], node_size=1.5, node_color='#336699',alpha=1, linewidths=0.2, font_size = 1.5)
          nx.draw_networkx_edges(G, pos[id], alpha=0.6,width=0.2)

        # plt.axis('off')
        # plt.title('Complete Graph of Odd-degree Nodes')
        # plt.show()
        plt.tight_layout()
        plt.savefig(fname + str(j) + '.png', dpi=800)
        plt.close()

import create_graphs
#import random
from random import shuffle
#from sklearn import preprocessing
import time
np.random.seed(seed=int(time.time()))
#np.random.seed(123)
#random.seed(123)

def test_create_community():
  num_surfaces = 18
  num_points = 400
  num_perm = 3
  #types=['caveman_2','caveman_4']
  graphs_create = create_graphs.create(create_graphs.Graph_Args('caveman_4'))
  np.random.shuffle(graphs_create)
  feature_graphs, pos, graphs = generate_feature_list(graphs_create, num_perm)
  feature_graphs[:], pos[:], graphs[:] = shuffle_list(list(feature_graphs), list(pos), list(graphs))
  
  #graphs = np.asarray(graphs, dtype=np.float32)
  for i in range(len(graphs)):
    graphs[i] = nx.to_numpy_matrix(graphs[i])
  graphs = np.array(graphs, dtype=np.float32)
  feature_graphs = np.array(feature_graphs, dtype=np.float32)

  print("graphs[0].shape: ", graphs[0].shape)
  print("feature_graphs[0].shape: ", feature_graphs[0].shape)
  counter = 0
  draw_graph(G_arr=graphs[0:40], row=2, col=2, pos=feature_graphs[0:40], fname='comm/comm_'+str(counter))
  
#test_create_community()

class GenerateDataGraph:

    def __init__(self, type_dataset='caveman_small', proportion=(0.8, 0.2), proportion_edge=[.8, .2], num_perm=10):

        graph_args = create_graphs.Graph_Args(type=type_dataset)
        graphs_create = create_graphs.create(graph_args)
        np.random.shuffle(graphs_create)
        self.num_nodes = graphs_create[0].number_of_nodes()
        self.num_edges = graphs_create[0].number_of_edges()
        feature_graphs, pos, graphs = generate_feature_list(graphs_create, num_perm)
        self.num_graphs = len(graphs)
        feature_graphs[:], pos[:], graphs[:] = shuffle_list(list(feature_graphs), list(pos), list(graphs))

        #feature_graphs = np.array(feature_graphs)
        #X_features = feature_graphs.reshape(-1,feature_graphs.shape[-1])
        #std_scale = preprocessing.StandardScaler().fit(X_features)
        #X_std = std_scale.transform(X_features)
        #feature_graphs = X_std.reshape(feature_graphs.shape[0],feature_graphs.shape[1],feature_graphs.shape[2])

        #another, identity, full
        input_graphs = self.generate_input_graphs('identity', self.num_graphs, self.num_nodes, proportion=proportion_edge)
        #('identity', self.num_graphs, self.num_nodes)
        self.num_features = feature_graphs[0].shape[-1]

        self.graphs_test = graphs[int(proportion[0] * self.num_graphs):] #0.8
        #save_graph_list(self.graphs_test, 'gt.dat')

        for i in range(self.num_graphs):
          graphs[i] = nx.to_numpy_matrix(graphs[i])

        n_training = 0.96 #0.8
        n_eval = 0.02 #0.1
        n_test = 0.02 #0.1
        graphs_test = graphs[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        graphs_train = graphs[0:int(self.num_graphs*n_training)] #0.8
        graphs_validate = graphs[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        feature_test = feature_graphs[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        feature_train = feature_graphs[0:int(self.num_graphs*n_training)] #0.8
        feature_validate = feature_graphs[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        input_graph_test = input_graphs[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        input_graph_train = input_graphs[0:int(self.num_graphs*n_training)] #0.8
        input_graph_validate = input_graphs[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        self.pos_test = pos[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        self.pos_train = pos[0:int(self.num_graphs*n_training)] #0.8
        self.pos_validate = pos[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        self.num_val = len(graphs_validate)
        self.num_test = len(graphs_test)
        self.num_training = len(graphs_train)

        self.train_generator = self.batch_generator(graphs_train, feature_train, input_graph_train)
        self.valid_generator = self.batch_generator(graphs_validate, feature_validate, input_graph_validate)
        self.test_generator = self.batch_generator(graphs_test, feature_test, input_graph_test)

        print("DATASET:", type_dataset)
        print("num_graphs:", self.num_graphs)
        print("num_nodes by graph:", self.num_nodes)
        print("num_edges by graph:", self.num_edges)
        print("num_features by node:", self.num_features)
        print("num_training:", self.num_training)
        print("num_val:", self.num_val)
        print("num_test:", self.num_test)

    def generate_input_graphs( self, type, num_graphs, num_nodes, proportion=[5./10, 5./10] ):
        inputs_graphs = []
        for i in range(num_graphs):
            if type == 'identity':
                graph_i = np.identity(num_nodes)#np.zeros(size=(num_nodes,num_nodes))
            elif type == 'full':
                graph_i = np.ones((num_nodes,num_nodes))
            else: #'aleatory' #p=propostion of 0's and 1's
                graph_i = np.random.choice([0, 1], size=(num_nodes,num_nodes), p=proportion)
                graph_i = graph_i * graph_i.T
            np.fill_diagonal(graph_i, 1.0)
            inputs_graphs.append(graph_i.astype(np.float32))
        return inputs_graphs

    def batch_generator( self, db_graph, db_feature, db_input_graph ):
        def gen_batch( batch_size ):
            for offset in range(0, len(db_graph), batch_size):
                files_graph = db_graph[offset:offset+batch_size]
                files_feature = db_feature[offset:offset+batch_size]
                files_input_graph = db_input_graph[offset:offset+batch_size]

                yield np.array( files_graph ), np.array( files_feature ), np.array( files_input_graph )
        return gen_batch

def save_graph(name, points_coord, adj, dim):
  '''points_coord.shape: (num_points, coord=3),
    adj.shape:(num_points,num_points)'''
  f_node = open(name+'_node.csv', 'w')
  x = points_coord[:,0]
  y = points_coord[:,1]
  if dim == 3:
    z = points_coord[:,2]

  # positions of nodes
  for i in range(len(x)):
    f_node.write(str(x[i]) + ", " + str(y[i])),
    if dim == 3:
      f_node.write(", " + str(z[i])),
    f_node.write("\n")
  f_node.close()

  # edges
  f_edge = open(name+'_edge.csv', 'w')
  for i in range(adj.shape[0]):
    for j in range(i,adj.shape[1]):
      if adj[i][j] and i != j:
        f_edge.write(str(x[i])+", "+str(y[i]))
        if dim == 3:
          f_edge.write(", "+str(z[i])),
        f_edge.write(", "+str(x[j])+", "+str(y[j]))
        if dim == 3:
          f_edge.write(", "+str(z[j])),
        f_edge.write("\n")
  f_edge.close()


def test_batch_gen():
  #['caveman_2', 'caveman_4']
  gen_graph = GenerateDataGraph(type_dataset='caveman_4', num_perm=10)
  epochs=1
  batch_size=40
  for epoch in range(epochs):
    print("\n########## epoch " + str(epoch+1) + " ##########")
    gen_trainig = gen_graph.train_generator( batch_size = batch_size )
    counter = 0
    for gt_graph, set_feature, in_graph in gen_trainig:
      print("---- batch ----")
      print("gt_graph.shape: ", gt_graph.shape)
      print("set_feature.shape: ", set_feature.shape)
      print("in_graph.shape: ", in_graph.shape)
      draw_graph(G_arr=gt_graph, row=2, col=2, pos=set_feature, fname='comm/comm_'+str(counter))

      #for k in range(len(gt_graph)):
      #  counter += 1
      #  save_graph(name='comm/comm_'+str(counter), points_coord=set_feature[k], adj=gt_graph[k], dim=2)
      break

test_batch_gen()

