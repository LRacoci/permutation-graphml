import networkx as nx
import numpy as np

#from utils import *
#from data import *

def caveman_special(c,k,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.connected_caveman_graph(c, k)#nx.caveman_graph(c, k)
    # remove 50% edges
    #p = 1-p_edge
    #for (u, v) in list(G.edges()):
    #    if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
    #        G.remove_edge(u, v)

    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(nx.connected_component_subgraphs(G), key=len)

    return G

class Graph_Args():
    def __init__(self, type='caveman', max_num_node=None, max_prev_node=None, noise=10):
      self.graph_type = type # Which dataset is used to train the model
      self.max_num_node = None # max number of nodes in a graph
      self.max_prev_node = None # max previous node that looks back
      self.noise = None
      ### Which dataset is used to train the model
      #self.graph_type = 'DD'
      #self.graph_type = 'caveman'
      #self.graph_type = 'caveman_small'
      #self.graph_type = 'caveman_small_single'
      #self.graph_type = 'community4'
      #self.graph_type = 'grid'
      #self.graph_type = 'grid_small'
      #self.graph_type = 'ladder_small'

      #self.graph_type = 'enzymes'
      #self.graph_type = 'enzymes_small'
      #self.graph_type = 'barabasi'
      #self.graph_type = 'barabasi_small'
      #self.graph_type = 'citeseer'
      #self.graph_type = 'citeseer_small'

      #self.graph_type = 'barabasi_noise'
      #self.noise = 10
      #
      # if self.graph_type == 'barabasi_noise':
      #     self.graph_type = self.graph_type+str(self.noise)

      # if none, then auto calculate
      #self.max_num_node = None# max number of nodes in a graph
      #self.max_prev_node = None # max previous node that looks back

def create(args):
### load datasets
    graphs=[]
    # synthetic graphs
    if args.graph_type=='ladder':
        graphs = []
        print("num nodes: ", 100, 201)
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='ladder_small':
        graphs = []
        for i in range(2, 11):
            graphs.append(nx.ladder_graph(i))
            print("len: ", len(graphs[-1]))
        args.max_prev_node = 10
    elif args.graph_type=='tree':
        graphs = []
        #print("num nodes: ", "i: ", 2,5, " j: ", 3,5)
        for i in range(2,5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    elif args.graph_type=='caveman':
        # graphs = []
        # for i in range(5,10):
        #     for j in range(5,25):
        #         for k in range(5):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                j = 80
                for k in range(10):
                    graphs.append(caveman_special(i,j, p_edge=0.3))
        args.max_prev_node = 100
    elif args.graph_type=='caveman_2':
      graphs = []
      i = 2
      j = 20
      for k in range(300):
          graphs.append(caveman_special(c=i,k=j, p_edge=0.3))
      args.max_prev_node = 100
    elif args.graph_type=='caveman_4':
      graphs = []
      i = 4
      j = 20
      for k in range(500):
          graphs.append(caveman_special(c=i,k=j, p_edge=0.3))
      args.max_prev_node = 100
    elif args.graph_type=='caveman_small':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                j = 10
                for k in range(20):
                    graphs.append(caveman_special(i, 20, p_edge=0.8)) # default 0.8
        args.max_prev_node = 20
    elif args.graph_type=='caveman_small_single':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graphs.append(caveman_special(i, j, p_edge=0.5))
        args.max_prev_node = 20
    elif args.graph_type.startswith('community'):
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        #c_sizes = [15] * num_communities
        for k in range(3000):
            graphs.append(n_community(c_sizes, p_inter=0.01))
        args.max_prev_node = 80
    elif args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            i = 16 #19
            for j in range(10,20):
                j = 16 #19
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    elif args.graph_type=='grid_small':
        graphs = []
        for i in range(2,5):
            for j in range(2,6):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 15
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
             i = 200
             for j in range(4,5):
                 for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 130
    elif args.graph_type=='barabasi_small':
        graphs = []
        for i in range(4,21):
             i = 20
             for j in range(3,4):
                 for k in range(40): #10
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 20
    elif args.graph_type=='grid_big':
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 90

    elif 'barabasi_noise' in args.graph_type:
        graphs = []
        for i in range(100,101):
            for j in range(4,5):
                for k in range(500):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        graphs = perturb_new(graphs,p=args.noise/10.0)
        args.max_prev_node = 99

    # real graphs
    elif args.graph_type == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 25
    elif args.graph_type == 'enzymes_small':
        graphs_raw = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        graphs = []
        for G in graphs_raw:
            if G.number_of_nodes()<=20:
                graphs.append(G)
        args.max_prev_node = 15
    elif args.graph_type == 'protein':
        graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
        args.max_prev_node = 80
    elif args.graph_type == 'DD':
        graphs = Graph_load_batch(min_num_nodes=250, max_num_nodes=500, name='DD',node_attributes=False,graph_labels=True) #min:100,max:500
        args.max_prev_node = 230
    elif args.graph_type == 'citeseer':
        adj, features, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        #tmp_100=0; tmp_200=0; tmp_300=0; tmp_400=0;
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            '''
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                if G_ego.number_of_nodes() >= 100:
                  tmp_100 += 1
                if G_ego.number_of_nodes() >= 200:
                  tmp_200 += 1
                if G_ego.number_of_nodes() >= 300:
                  tmp_300 += 1
                if G_ego.number_of_nodes() >= 400:
                  tmp_400 += 1
            '''
            num_nodes=200
            if G_ego.number_of_nodes() >= num_nodes:
                degrees = list(G_ego.degree())
                sort_by_degree =sorted(degrees, key=lambda x: x[1])
                for i in range(len(sort_by_degree)-num_nodes):
                  G_ego.remove_node(sort_by_degree[i][0])
                for i in range(len(sort_by_degree)-num_nodes,len(sort_by_degree)):
                  #G_ego.remove_node(sort_by_degree[i][0])
                  G_ego.add_node(sort_by_degree[i][0], feature = features[sort_by_degree[i][0]])
                graphs.append(G_ego)
        args.max_prev_node = 250
        #print("tmp_100: ", tmp_100);print("tmp_200: ", tmp_200);print("tmp_300: ", tmp_300);print("tmp_400: ", tmp_400);
    elif args.graph_type == 'citeseer_small':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        shuffle(graphs)
        graphs = graphs[0:200]
        args.max_prev_node = 15

    return graphs
