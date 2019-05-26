#@title Imports  { form-width: "20%" }

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)

#@title Debug { form-width: "20%" }
debug_tags = {
    "", "darwin_batches_to_networkx_graphs"
}

import json
def json_default(obj) :
    from networkx.readwrite import json_graph
    class_name = obj.__class__.__name__
    serialization = {
        'DiGraph' : json_graph.adjacency_data,
        'int64' : int,
        'int32' : int,
        'float32' : float,
        'ndarray' : list
    }
    if class_name in serialization:
        return serialization[class_name](obj)
    else:
        msg = "Unserializable object {} of type '{}', add class '{}' to rules".format(obj, type(obj),class_name)
        print(msg)
        raise TypeError(msg)

def dumps(obj, indent = 4, default = json_default) :
    return json.dumps(obj, indent = indent, default = default)

def debug(obj, tag=""):
    global debug_tags
    if tag in debug_tags:
        print(tag, dumps(obj))

#@title Converters to/from Darwin's format  { form-width: "20%" }
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

def darwin_batches_to_networkx_graphs(adjs_gt, node_features, adjs_inp, set_segmentation):
    '''
    Args:
        adjs_gt : adjacency matrix of ground truth, shape = (num_graphs, num_nodes, num_nodes)
        node_features :  Matrix of node features, shape = (num_graphs, num_nodes, num_node_features)
        adjs_inp : adjacency matrix of input graph, shape = (num_graphs, num_nodes, num_nodes)

    Returns:
        graphs_tuple : GraphTuple from graph_nets
    '''
    nxGraphs = []
    for adj_gt, node_feature, adj_inp, set_segm in zip(adjs_gt, node_features, adjs_inp,set_segmentation):
        nxGraph = nx.from_numpy_matrix(adj_inp, create_using=nx.DiGraph)
        #Nodes
        nx.set_node_attributes(G = nxGraph, name ="rgbxy", values = {
            n : val 
            for n,val in enumerate(node_feature)
        })
        set_segm = set_segm[
            np.uint32(node_feature[:,-2]), 
            np.uint32(node_feature[:,-1])
        ]
        nx.set_node_attributes(G = nxGraph, name ="resp", values = {
            n : val
            for n,val in enumerate(set_segm)
        })
        #Edges
        nx.set_edge_attributes(G = nxGraph, name ="resp", values = {
            (u,v) : adj_gt[u][v]
            for (u,v) in nxGraph.edges
        })
        debug(nxGraph, "darwin_batches_to_networkx_graphs")
        nxGraphs.append(nxGraph)

    return nxGraphs

def graphs_tuple_dumps(graphs_tuple):
    data_dicts = utils_np.graphs_tuple_to_data_dicts(graphs_tuple)
    return json_dumps(data_dicts)

def graphs_tuple_loads(string_dump):
    data_dicts = json.loads(string_dump)
    for data_dict in data_dicts:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

    graphs_tuple = utils_np.data_dicts_to_graphs_tuple(data_dicts)
    return graphs_tuple


def graphs_tuples_to_darwin_batches(graph_nets):
    '''
    Args:
        graphs_tuple : GraphTuple from graph_nets
    Returns:
        graph : adjacency matrix of the graph, shape = (num_graphs, num_nodes, num_nodes)
        node_features :  Matrix of node features, shape = (num_graphs, num_nodes, num_node_features)
    '''
    adjs_gt = []
    node_features = []
    adjs_inp = []
    
    data_dicts = utils_np.graphs_tuple_to_data_dicts(graph_nets)
    for data_dict in data_dicts:
        nodes = data_dict['nodes']
        num_nodes= len(nodes)
        
        adj_inp = np.zeros(shape = (num_nodes, num_nodes))
        adj_gt = np.zeros(shape = (num_nodes, num_nodes))
        
        senders = data_dict['senders']
        receivers = data_dict['receivers']
        edges = data_dict['edges']
        
        adj_inp[senders, receivers] = 1.0
        adjs_gt[senders, receivers] = edges
        
        adjs_inp.append(adj_inp)
        adjs_gt.append(adj_gt)
        node_features.append(nodes)
    
    return np.array(adjs_gt), np.array(node_features), np.array(adjs_inp)

#@title Darwin's Images { form-width: "20%" }
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import tensorflow as tf
np.random.seed(123)

epsilon = 1e-12

class GenerateAdjMatrx:

    def __init__( self, type_dist="D4" , dim_x = 10, dim_y = 10 ):
        self.type_dist = type_dist
        self.img_h = dim_x
        self.img_w = dim_y
        '''
        distance D4 or city-block
                h-1,w
        h,w-1    h,w    h,w+1
                h+1,w
        '''
        self.D4_h = np.array([ 0,  0, 1, -1 ])
        self.D4_w = np.array([ -1, 1, 0,  0 ])

        '''
        distance D8 or chessboard
        h-1,w-1   h-1,w   h-1,w+1
        h,w-1     h,w     h,w+1
        h+1,w-1   h+1,w   h+1,w+1
        '''
        self.D8_h = np.array([ -1,  0,  1, -1, 1, -1, 0, 1 ])
        self.D8_w = np.array([ -1, -1, -1,  0, 0,  1, 1, 1 ])


    def adjmatrx_generator( self, dim_x = 10, dim_y = 10 ):
        self.img_h = dim_x
        self.img_w = dim_y

        len_dist = 0
        dist_h = []
        dist_w = []
        num_nodes = self.img_h * self.img_w
        self.adjmatrx = np.zeros( ( num_nodes, num_nodes ), dtype = np.float32 )

        if self.type_dist == "D4":
            len_dist = 4
            dist_h = self.D4_h
            dist_w = self.D4_w
        elif self.type_dist == "D8":
            len_dist = 8
            dist_h = self.D8_h
            dist_w = self.D8_w
        else:
            pass

        for node in range( num_nodes ):
            h = int(node / self.img_w)
            w = node % self.img_w
            self.adjmatrx[ node, node ] = 1.0
            for k in range(len_dist):
                hi = h + dist_h[ k ]
                wi = w + dist_w[ k ]
                if hi >= 0 and hi < self.img_h and wi >= 0 and wi < self.img_w:
                    if np.random.randint(2) == 0:
                        self.adjmatrx[ node, int(self.img_w * hi + wi) ] = 1.0
                        self.adjmatrx[ int(self.img_w * hi + wi), node ] = 1.0

        return np.copy(self.adjmatrx.astype( np.float32 ))

    def adjmatrx_generator_batch_random( self, num_batch, dim_x = 10, dim_y = 10 ):
        self.img_h = dim_x
        self.img_w = dim_y

        len_dist = 0
        dist_h = []
        dist_w = []
        num_nodes = self.img_h * self.img_w
        self.adjmatrx = np.zeros( ( num_batch, num_nodes, num_nodes ), dtype = np.float32 )

        if self.type_dist == "D4":
            len_dist = 4
            dist_h = self.D4_h
            dist_w = self.D4_w
        elif self.type_dist == "D8":
            len_dist = 8
            dist_h = self.D8_h
            dist_w = self.D8_w
        else:
            pass

        for n_batch in range(num_batch):
            for node in range( num_nodes ):
                h = int(node / self.img_w)
                w = node % self.img_w
                self.adjmatrx[ n_batch, node, node ] = 1.0
                for k in range(len_dist):
                    hi = h + dist_h[ k ]
                    wi = w + dist_w[ k ]
                    if hi >= 0 and hi < self.img_h and wi >= 0 and wi < self.img_w:
                        self.adjmatrx[ n_batch, node, int(self.img_w * hi + wi) ] = 1.0
                        self.adjmatrx[ n_batch, int(self.img_w * hi + wi), node ] = 1.0

        return np.copy(self.adjmatrx.astype( np.float32 ))

    def adjmatrx_groundthuth( self, img_groundthuth ):
        self.img_h = img_groundthuth.shape[ 0 ]
        self.img_w = img_groundthuth.shape[ 1 ]

        len_dist = 0
        dist_h = []
        dist_w = []
        num_nodes = self.img_h * self.img_w
        self.adjmatrx_gt = np.zeros( ( num_nodes, num_nodes ), dtype = np.float32 )

        if self.type_dist == "D4":
            len_dist = 4
            dist_h = self.D4_h
            dist_w = self.D4_w
        elif self.type_dist == "D8":
            len_dist = 8
            dist_h = self.D8_h
            dist_w = self.D8_w
        else:
            pass

        for node in range( num_nodes ):
            h = int(node / self.img_w)
            w = node % self.img_w
            self.adjmatrx_gt[ node, node ] = 1.0
            for k in range(len_dist):
                hi = h + dist_h[ k ]
                wi = w + dist_w[ k ]
                if hi >= 0 and hi < self.img_h and wi >= 0 and wi < self.img_w:
                    if( img_groundthuth[ h, w ] == img_groundthuth[ hi, wi ] ):
                        self.adjmatrx_gt[ node, int(self.img_w * hi + wi) ] = 1.0
                        self.adjmatrx_gt[ int(self.img_w * hi + wi), node ] = 1.0

        return np.copy(self.adjmatrx_gt)


    def adjmatrx_loss( self, adj_groundthuth, adj_prediction, dim_x = 10, dim_y = 10 ):
        self.img_h = dim_x
        self.img_w = dim_y

        len_dist = 0
        dist_h = []
        dist_w = []
        num_nodes = self.img_h * self.img_w
        self.adj_loss = 0.0

        if self.type_dist == "D4":
            len_dist = 4
            dist_h = self.D4_h
            dist_w = self.D4_w
        elif self.type_dist == "D8":
            len_dist = 8
            dist_h = self.D8_h
            dist_w = self.D8_w
        else:
            pass

        n = 0.0
        for node in range( num_nodes ):
            h = int(node / self.img_w)
            w = node % self.img_w

            for k in range(len_dist):
                hi = h + dist_h[ k ]
                wi = w + dist_w[ k ]
                if hi >= 0 and hi < self.img_h and wi >= 0 and wi < self.img_w:
                    n += 1
                    value_pred = adj_prediction[ node, int(self.img_w * hi + wi) ]
                    value_ground = adj_groundthuth[ node, int(self.img_w * hi + wi) ]
                    self.adj_loss += np.abs( value_ground - value_pred )

        self.adj_loss /= n
        return self.adj_loss


class GenerateImg:
    def __init__( 
            self, 
            dim_x = 10, 
            dim_y = 10, 
            type_dist="D4", 
            proportion=(0.05, 0.2, 1000), 
            option_shape='all', 
            color_rand = True, 
            noise_data = True 
    ):
        ''' proportion (validation, test, rest is training) '''
        self.img_h = dim_x
        self.img_w = dim_y
        self.num_val = int(proportion[0] * proportion[2])
        self.num_test = int(proportion[1] * proportion[2])
        self.num_training = int(proportion[2] - self.num_test)
        self.option_shape = option_shape
        self.color_rand = color_rand
        self.noise_data = noise_data
        self.type_dist = type_dist

    def func_perm(self, img_bgr_, img_ground_truth_, label_all_, A_gt_):
        id_perm = np.random.permutation(self.img_h*self.img_w)
        #id_perm_w = np.random.permutation(self.img_w)
        A_gt = np.zeros_like(A_gt_)
        for i in range(A_gt.shape[0]):
            for j in range(A_gt.shape[1]):
                if A_gt[i][j] == 1. or A_gt[j][i] == 1.:
                    A_gt[id_perm[i]][id_perm[j]] = 1.#A_gt[i][j]
                    A_gt[id_perm[j]][id_perm[i]] = 1.#A_gt[i][j]

        img_bgr = np.zeros((
                img_bgr_.shape[0],
                img_bgr_.shape[1], 
                img_bgr_.shape[2]+2
        ))
        img_ground_truth = np.zeros_like(img_ground_truth_)
        label_all = np.zeros_like(label_all_)
        for node in range( A_gt.shape[0] ):
            h = int(node / self.img_w)
            w = node % self.img_w
            h_perm = int(id_perm[node] / self.img_w)
            w_perm = id_perm[node] % self.img_w

            img_bgr[h_perm][w_perm] = img_bgr[h][w]
            img_bgr[h_perm][w_perm][3] = h
            img_bgr[h_perm][w_perm][4] = w
            img_ground_truth[h_perm][w_perm] = img_ground_truth[h][w]
            label_all[h_perm][w_perm] = label_all[h][w]

        return img_bgr, img_ground_truth, label_all, A_gt

    def add_position(self, img_bgr_):
        img_bgr = np.zeros((img_bgr_.shape[0],img_bgr_.shape[1], img_bgr_.shape[2]+2))
        for h in range(img_bgr_.shape[0]):
            for w in range(img_bgr_.shape[1]):
                img_bgr[h][w][0:3] = img_bgr_[h][w]
                img_bgr[h][w][3] = h
                img_bgr[h][w][4] = w
        return img_bgr

    def load_data(self):
        sample_img_train = []
        sample_label_train = []
        sample_label_split_train = []
        sample_A_gt_train = []
        sample_B_in_train = []
        for n_train in range(self.num_training):
            #img_bgr.shape = (10, 10, 3)
            img_bgr, img_ground_truth, label_all, A_gt, B_in = self.generate_syntetic_data()
            img_bgr = self.add_position(img_bgr)

            sample_img_train.append(img_bgr)
            sample_label_train.append(img_ground_truth)
            sample_label_split_train.append(label_all)
            sample_A_gt_train.append(A_gt)
            sample_B_in_train.append(B_in)

        sample_img_val = []
        sample_label_val = []
        sample_label_split_val = []
        sample_A_gt_val = []
        sample_B_in_val = []
        for n_val in range(self.num_val):
            img_bgr, img_ground_truth, label_all, A_gt, B_in = self.generate_syntetic_data()
            img_bgr = self.add_position(img_bgr)

            sample_img_val.append(img_bgr)
            sample_label_val.append(img_ground_truth)
            sample_label_split_val.append(label_all)
            sample_A_gt_val.append(A_gt)
            sample_B_in_val.append(B_in)

        sample_img_test = []
        sample_label_test = []
        sample_label_split_test = []
        sample_A_gt_test = []
        sample_B_in_test = []
        for n_test in range(self.num_test):
            img_bgr, img_ground_truth, label_all, A_gt, B_in = self.generate_syntetic_data()
            img_bgr = self.add_position(img_bgr)

            sample_img_test.append(img_bgr)
            sample_label_test.append(img_ground_truth)
            sample_label_split_test.append(label_all)
            sample_A_gt_test.append(A_gt)
            sample_B_in_test.append(B_in)

        self.train_generator = self.batch_generator(
                sample_img_train, 
                sample_label_train, 
                sample_label_split_train, 
                sample_A_gt_train,
                sample_B_in_train
        )
        self.valid_generator = self.batch_generator(
                sample_img_val, 
                sample_label_val, 
                sample_label_split_val, 
                sample_A_gt_val,
                sample_B_in_val
        )
        self.test_generator = self.batch_generator(
                sample_img_test, 
                sample_label_test, 
                sample_label_split_test, 
                sample_A_gt_test,
                sample_B_in_test
        )

    def generate_color(self, color_rand):
        if( color_rand ):
            b_r_color = np.random.randint( 0, 30 ) #(0-30)
            b_g_color = np.random.randint( 0, 30 ) #(0-30)
            b_b_color = np.random.randint( 0, 164 ) + 90 #(90-254)

            r_r_color = np.random.randint( 0, 104 ) + 150 #(150-254)
            r_g_color = np.random.randint( 0, 30 ) #(0-30)
            r_b_color = np.random.randint( 0, 10 ) #(0-10)
            return [ b_r_color, b_g_color, b_b_color ], [ r_r_color, r_g_color, r_b_color ]
        else:
            r_color = [ 187.0, 5.0, 13.0 ]
            b_color = [ 51.0, 2.0, 151.0 ]
            return b_color, r_color

    def linear_function( self, x1, y1, x2, y2, xi, yi ):
        y = ( ( ( y2 - y1 ) / ( x2 - x1 + 0.001 ) ) * ( xi - x1 ) ) + y1
        if yi >= y:
            return True
        else:
            return False

    def point_inside_circle( self, x, y, r, xi, yi ):
        if ( (xi - x)*(xi - x) + (yi - y)*(yi - y) <= r*r):
            return True;
        else:
            return False;

    def point_inside_rectangle( self, x1, y1, x2, y2, xi, yi ):
        if ( ( xi >= x1 and xi <= x2 ) and ( yi >= y1 and yi <= y2 ) ):
            return True;
        else:
            return False;

    def generate_line( self, img_ground_truth, img, class_blue, class_red ):
        x1 = np.random.randint( 0, self.img_h - 1 ) * 1.0
        y1 = np.random.randint( 0, self.img_w - 1 ) * 1.0
        while True:
            x2 = np.random.randint( -50, 50 ) * 1.0
            y2 = np.random.randint( -50, 50 ) * 1.0
            if x1 != x2 or y1 != y2:
                    break

        if( self.noise_data == False ):
            color_set_blue, set_color_red = self.generate_color(self.color_rand)

        for i in range( img.shape[ 0 ] ):
            for j in range( img.shape[ 1 ] ):
                if( self.noise_data == True ):
                    color_set_blue, set_color_red = self.generate_color(self.color_rand)

                if self.linear_function( x1, y1, x2, y2, i, j ):
                    img[ i, j ] = color_set_blue #R,G,B
                    class_blue[ i, j ] = 1.0
                else:
                    img_ground_truth[ i, j ] = 1.0
                    #R,G,B
                    img[ i, j ] = set_color_red 
                    class_red[ i, j ] = 1.0

    def generate_circle( self, img_ground_truth, img, class_blue, class_red ):
        c_x = np.random.randint( 0, self.img_h - 1 ) * 1.0
        c_y = np.random.randint( 0, self.img_w - 1 ) * 1.0
        r = np.random.randint( 0, min( self.img_h/2, self.img_h/2 ) ) * 1.0

        # blue:0, red:1
        color_square = np.random.randint( 0, 1 ) 

        if( self.noise_data == False ):
            color_set_blue, set_color_red = self.generate_color(self.color_rand)

        for i in range( img.shape[ 0 ] ):
            for j in range( img.shape[ 1 ] ):
                if( self.noise_data == True ):
                    color_set_blue, set_color_red = self.generate_color(self.color_rand)

                if self.point_inside_circle( c_x, c_y, r, i, j ):
                    # square blue
                    if color_square == 0: 
                        #R,G,B
                        img[ i, j ] = color_set_blue 
                        class_blue[ i, j ] = 1.0
                    else: # square red
                        img_ground_truth[ i, j ] = 1.0
                        #R,G,B
                        img[ i, j ] = set_color_red 
                        class_red[ i, j ] = 1.0
                else:
                    # brackground red
                    if color_square == 0: 
                        img_ground_truth[ i, j ] = 1.0
                        #R,G,B
                        img[ i, j ] = set_color_red 
                        class_red[ i, j ] = 1.0
                # brackground blue
                    else: 
                        #R,G,B
                        img[ i, j ] = color_set_blue 
                        class_blue[ i, j ] = 1.0

    def generate_rectangle( self, img_ground_truth, img, class_blue, class_red ):
        x1 = np.random.randint( 0, self.img_h - 1 ) * 1.0
        y1 = np.random.randint( 0, self.img_w - 1 ) * 1.0
        while True:
            x2 = np.random.randint( -50, 50 ) * 1.0
            y2 = np.random.randint( -50, 50 ) * 1.0
            if x1 != x2 or y1 != y2:
                break

        x_min = min(x1, x2); x_max = max(x1, x2)
        y_min = min(y1, y2); y_max = max(y1, y2)

        # blue:0, red:1
        color_square = np.random.randint( 2 ) 

        if( self.noise_data == False ):
            color_set_blue, set_color_red = self.generate_color(self.color_rand)

        for i in range( img.shape[ 0 ] ):
            for j in range( img.shape[ 1 ] ):
                if( self.noise_data == True ):
                    color_set_blue, set_color_red = self.generate_color(self.color_rand)

                if self.point_inside_rectangle( x_min, y_min, x_max, y_max, i, j ):
                    # square blue
                    if color_square == 0: 
                        #R,G,B (51.0, 2.0, 151.0)
                        img[ i, j ] = color_set_blue 
                        class_blue[ i, j ] = 1.0
                    # square red
                    else: 
                        img_ground_truth[ i, j ] = 1.0
                        #R,G,B (187.0, 5.0, 13.0)
                        img[ i, j ] = set_color_red 
                        class_red[ i, j ] = 1.0
                else:
                    # brackground red
                    if color_square == 0: 
                        img_ground_truth[ i, j ] = 1.0
                        #R,G,B
                        img[ i, j ] = set_color_red 
                        class_red[ i, j ] = 1.0
                    # brackground blue
                    else: 
                        #R,G,B
                        img[ i, j ] = color_set_blue 
                        class_blue[ i, j ] = 1.0

    def generate_syntetic_data( self ):
        """ option_shape=[ 'all', 'line', 'circle', 'rectangle'] """
        label_list = []
        img_ground_truth = np.zeros( ( self.img_h, self.img_w ), dtype = np.float32 )
        img = np.zeros( ( self.img_h, self.img_w, 3 ), dtype = np.float32 )
        class_blue = np.zeros( ( self.img_h, self.img_w ), dtype = np.float32 )
        class_red = np.zeros( ( self.img_h, self.img_w ), dtype = np.float32 )

        '''line, square, grill, rectangle, cross'''
        if self.option_shape == 'line':
            sample_type = 0
        elif self.option_shape == 'circle':
            sample_type = 1
        elif self.option_shape == 'rectangle':
            sample_type = 2
        else: 
            sample_type = np.random.randint( 3 ) #0,1,2

        if sample_type == 0:
            self.generate_line( img_ground_truth, img, class_blue, class_red )
        elif sample_type == 1:
            self.generate_circle( img_ground_truth, img, class_blue, class_red )
        else:
            self.generate_rectangle( img_ground_truth, img, class_blue, class_red )

        label_list.append( class_blue )
        label_list.append( class_red )
        # 2 classes generates
        label_all = np.dstack( label_list ).astype( np.float32 ) 
        r, g, b = cv2.split( img )
        img_bgr = cv2.merge( [ b, g, r ] )

        gen_adj = GenerateAdjMatrx( type_dist = self.type_dist )
        A_gt = gen_adj.adjmatrx_groundthuth( img_ground_truth )

        B_in = gen_adj.adjmatrx_groundthuth(img_ground_truth * 0)

        return img_bgr, img_ground_truth, label_all, A_gt, B_in

    def batch_generator( self, db_img, db_label, db_label_split, db_A_gt, db_B_in):
        def gen_batch( batch_size ):
            for offset in range(0, len(db_img), batch_size):
                files_img = db_img[offset:offset+batch_size]
                files_label = db_label[offset:offset+batch_size]
                files_label_split = db_label_split[offset:offset+batch_size]
                files_A_gt = db_A_gt[offset:offset+batch_size]
                files_B_in = db_B_in[offset:offset+batch_size]

                yield tuple([
                        np.array(files_A_gt), 
                        np.array(files_img).reshape(len(files_img),self.img_h*self.img_w,-1), 
                        np.array( files_label ), 
                        np.array(files_B_in)
                ])
        return gen_batch



#@title Graph Plot Helper Class  { form-width: "30%" }

class CDisplay:

    def display_images( self, img, label, label_by_classes, name ):
        b, g, r = cv2.split( img )
        img_rgb = cv2.merge( [ r, g, b ] )

        fig = plt.figure()

        plt.subplot( 2, 2, 1 )
        plt.title('image, X', fontsize=9)
        plt.imshow( img_rgb.astype(np.uint8) )

        plt.subplot( 2, 2, 2 )
        plt.title('label, Y', fontsize=9)
        plt.imshow( label )

        plt.subplot( 2, 2, 3 )
        plt.title('Class blue', fontsize=9)
        plt.imshow( label_by_classes[ :, :, 0 ].astype(np.uint8) )

        plt.subplot( 2, 2, 4 )
        plt.title('Class red', fontsize=9)
        plt.imshow( label_by_classes[ :, :, 1 ].astype(np.uint8) )

        plt.show()
        fig.savefig( name, dpi = fig.dpi )

    def display_results( self, img, label, pred, name ):
        b, g, r = cv2.split( img )
        img_rgb = cv2.merge( [ r, g, b ] )

        fig = plt.figure()

        plt.subplot( 1, 3, 1 )
        plt.title('image, X', fontsize=9)
        plt.imshow( img_rgb.astype(np.uint8) )

        plt.subplot( 1, 3, 2 )
        plt.title('label, Y', fontsize=9)
        plt.imshow( label.astype(np.uint8) )

        plt.subplot( 1, 3, 3 )
        plt.title('prediction', fontsize=9)
        plt.imshow( pred.astype(np.uint8) )

        plt.show()
        fig.savefig( name, dpi = fig.dpi )

    def display_neighborhood(
            self, 
            img_orig, 
            img_pred, 
            adj_original, 
            adj_update,
            img_h, 
            img_w, 
            name 
    ):

        fig = plt.figure()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        ax1.set_title('Img Original', fontsize=9)
        ax1.imshow( img_orig.astype(np.uint8) )

        ax2.set_title('Img Prediction', fontsize=9)
        ax2.imshow( img_pred.astype(np.uint8) )

        ch = np.arange(img_h)
        cw = np.arange(img_w)
        xx, yy = np.meshgrid(ch, cw)
        ax1.plot(xx,yy, 'o', c='w', markersize=1)
        ax2.plot(xx,yy, 'o', c='r', markersize=1)


        for h in range(img_h * img_w):
            for w in range(h, img_h * img_w):
                if( h != w ):
                    hi = [ int(h / img_w), int(w / img_w) ]
                    wi = [ int(h % img_w), int(w % img_w) ]

                    if adj_original[h,w] >= epsilon:
                        ax1.plot(
                            wi, 
                            hi, 
                            linewidth=0.7, 
                            color='w', 
                            linestyle='-',
                            alpha=adj_original[h,w], 
                            marker='o', 
                            markersize=1.0 
                        )
                    if (adj_update[h,w] >= epsilon):
                        ax2.plot(
                            wi, 
                            hi, 
                            linewidth=0.7, 
                            color='r', 
                            linestyle='-',
                            alpha = round( adj_update[h,w], 2 ) 
                        )

        fig.savefig( name, dpi = 400 ) #dpi = fig.dpi,

    def display_neighborhood2( 
        self, 
        img_orig, 
        img_pred, 
        adj_original, 
        adj_update, 
        img_h, 
        img_w, 
        name 
    ):
    '''Show the graph using the gt classifier as background(img_orig)'''
    fig = plt.figure()

    plt.imshow(img_orig)

    h = []
    w = []
    for i in range(img_h):
        for j in range(img_w):
            h.append(i)
            w.append(j)

    plt.plot(h,w, 'o', c='r', markersize=3)
    for h in range(img_h * img_w):
        for w in range(h, img_h * img_w):
            if (adj_update[h,w] >= epsilon):
                hi = np.array([int(h / img_w), int(w / img_w)],dtype=np.int8)
                wi = np.array([ int(h % img_w), int(w % img_w)],dtype=np.int8)
                plt.plot(
                    wi, 
                    hi, 
                    linewidth=2.0, 
                    color='r', 
                    linestyle='-',
                    alpha = round( adj_update[h,w], 2 )
                )

    fig.savefig( name, dpi = 100 ) #dpi = fig.dpi,

    def displayAdjMatrix( self, adj_update, name ):
        fig = plt.figure()
        plt.imshow( adj_update )
        plt.show()
        fig.savefig( name, dpi = fig.dpi )

def test_batch_gen():
    #------------------ Geometric shape synthetic data ------------------
    num_samples = 7
    num_points = 9 #square

    dim_h = int(np.sqrt(num_points))
    dim_w = int(np.sqrt(num_points))
    #num_data = 1000
    display = CDisplay()
    #synthetic data
    gen_dataset = GenerateImg(
            dim_x = dim_h, 
            dim_y = dim_w, 
            proportion=(0.05, 0.2, num_samples) 
    )
    gen_dataset.load_data()

    epochs=1
    batch_size=3
    for epoch in range(epochs):
        print("\n########## epoch " + str(epoch+1) + " ##########")
        gen_trainig = gen_dataset.train_generator( batch_size = batch_size )
        counter = 0
        for gt_graph, set_feature, set_segmentation, in_graph in gen_trainig:
            print("---- batch ----")
            print("gt_graph.shape: ", gt_graph.shape)
            print(gt_graph)
            print("set_feature.shape: ", set_feature.shape)
            print(set_feature)
            print("set_segmentation.shape: ", set_segmentation.shape)
            print(set_segmentation)
            print("in_graph.shape: ", in_graph.shape)
            print(in_graph)
            
            nxGraphs = darwin_batches_to_networkx_graphs(gt_graph, set_feature, in_graph, set_segmentation)
            
            #display = geometric_shape_dataset.CDisplay()
            shape_img = set_feature.shape
            dim_h, dim_w = int(np.sqrt(shape_img[1])), int(np.sqrt(shape_img[1]))
            img_set_feature = set_feature.reshape(
                    shape_img[0], 
                    dim_h, 
                    dim_w, 
                    shape_img[2]
            )
            img_set_feature = img_set_feature[:,:,:,0:3]
            for k in range(gt_graph.shape[0]):
                display.display_neighborhood(
                        img_set_feature[k], 
                        set_segmentation[k],
                        in_graph[k], gt_graph[k], 
                        dim_h, 
                        dim_w, 
                        'geo/img_'+str(counter)+'.png'
                ) #the second gt_graph will be the prediction
                counter += 1
            break

test_batch_gen()