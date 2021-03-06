# -*- coding: utf-8 -*-
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

SEED = 200
np.random.seed(SEED)
tf.set_random_seed(SEED)

!mkdir geo

#@title Debug { form-width: "20%" }
debug_tags = {
    "create_placeholders",
#     "raws", 
#     "source", 
#     "target",
    ""
}

import json
def json_default(obj) :
    from networkx.readwrite import json_graph
    class_name = obj.__class__.__name__
    serialization = {
        "Tensor" : lambda t : {
            "name" : str(t.name),
            "shape": str(t.shape),
            "dtype" : str(t.dtype)
            #Tensor("placeholders_from_networkxs/nodes:0", shape=(?, 6), dtype=float64)
        },
        "Operation" : lambda o : {
            "__dict__" : o.__dict__
        },
        'DiGraph' : json_graph.adjacency_data,
        'int64' : int,
        'int32' : int,
        'float32' : float,
        'ndarray' : list
    }
    if class_name in serialization:
        return serialization[class_name](obj)
    
    return repr(obj)
    
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
        nx.set_node_attributes(G = nxGraph, name ="solution", values = {
            n : val
            for n,val in enumerate(set_segm)
        })
        #Edges
        nx.set_edge_attributes(G = nxGraph, name ="solution", values = {
            (u,v) : adj_gt[u][v]
            for (u,v) in nxGraph.edges
        })
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

        #plt.colorbar(mappable=ax1, cax=None, ax=None)
        #synt_data = Parallel(n_jobs=-1)( \
        #    delayed(generate_syntetic_data)(self.img_h, self.img_w) for bt in range( batch_size ))

        #Parallel(n_jobs=3)( delayed(self.display_adj_mtrx)( img_h, img_w, h, w, ax1, adj_original[h,w] ) for w in range(img_h * img_w) for h in range(img_h * img_w) )
        #display_adj_mtrx( img_h, img_w, h, w, ax1, value_adj_original )

        #-- for h in range(img_h * img_w):
        #--     for w in range(img_h * img_w):
        #--         self.display_adj_mtrx( img_h, img_w, h, w, ax1, adj_original[h,w] )
        #---ch = []
        #---cw = []
        #---for i in range(img_h):
        #---    for j in range(img_w):
        #---        #ax1.plot(i,j, 'o', c='r', markersize=4)
        #---        #ax2.plot(i,j, 'o', c='r', markersize=4)
        #---        ch.append(i)
        #---        cw.append(j)
        #---ax1.plot(ch,cw, 'o', c='r', markersize=4)
        #---ax2.plot(ch,cw, 'o', c='r', markersize=4)

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
                                #circle_t1 = plt.Circle((h, w), 0.1, color='r')
                                #circle_t2 = plt.Circle((w, h), 0.1, color='r')
                                #ax1.add_artist(circle_t1)
                                #ax1.add_artist(circle_t2)
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

        #----ax2.set_title('Img Prediction', fontsize=9)
        #----ax2.imshow( img_pred.astype(np.uint8) )
        #----for h in range(img_h * img_w):
                #----for w in range(h, img_h * img_w):
                        #----hi = [ int(h / img_w), int(w / img_w) ]
                        #----wi = [ int(h % img_w), int(w % img_w) ]
                        #circle_t1 = plt.Circle((h, w), 0.1, color='r')
                        #circle_t2 = plt.Circle((w, h), 0.1, color='r')
                        #ax2.add_artist(circle_t1)
                        #ax2.add_artist(circle_t2)
                        #if ( adj_update[h,w] < 1e-12 ): print ( adj_update[h,w] )
                        #abc = (adj_update[h,w]*10).astype(int)/10.0
                        #print ("-------> ", abc )
                        #----if (adj_update[h,w] >= epsilon):
                                #print ("adj_update[h,w]: ", round( adj_update[h,w], 2 ) )
                                #value_color = int(adj_update[h,w]*10)#min(1,int(adj_update[h,w]*10))
                                #print( "--> ", value_color )
                                #ax2.plot(hi, wi, linewidth=2.0, color=colors[value_color], linestyle='-')
                                #----ax2.plot(wi, hi, linewidth=2.0, color='r', linestyle='-', \
                                        #----alpha = round( adj_update[h,w], 2 ) )

        #plt.scatter(data2_x, data2_y, marker='s', c=data2[data2_x, data2_y])
        #for h in range(img_h):
        #    for w in range(img_w):
        #        circle_t = plt.Circle((h, w), 0.1, color='r')
        #        ax1.add_artist(circle_t)
        #        ax2.add_artist(circle_t)

        #plt.show()
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
        #plt.figure()
        #fig = plt.imshow(img_orig)
        plt.imshow(img_orig)
        #plt.imshow(data1, interpolation='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        h = []
        w = []
        for i in range(img_h):
                for j in range(img_w):
                        h.append(i)
                        w.append(j)
        #plt.scatter(h, w, marker='o', c='r', markersize=0.2)
        plt.plot(h,w, 'o', c='r', markersize=3)
        for h in range(img_h * img_w):
                for w in range(h, img_h * img_w):
                        if (adj_update[h,w] >= epsilon):
                                hi = np.array([int(h / img_w), int(w / img_w)],dtype=np.int8)
                                wi = np.array([ int(h % img_w), int(w % img_w)],dtype=np.int8)
                                plt.plot(wi, hi, linewidth=2.0, color='r', linestyle='-', \
                                        alpha = round( adj_update[h,w], 2 ))
        #---plt.plot([[1, 2], [2, 5]],[[5, 1], [3, 7]], linewidth=2.0, color='r', linestyle='-', alpha=0.5)
        fig.savefig( name, dpi = 100 ) #dpi = fig.dpi,

    def displayAdjMatrix( self, adj_update, name ):
        fig = plt.figure()
        plt.imshow( adj_update )#.astype(np.uint8) )
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

#@title Helper functions  { form-width: "20%" }

# pylint: disable=redefined-outer-name

def print_graph(g):
    for s, t, w in g.edges(data=True):
        if 'features' not in w:
            print(s, t, w, "(problem)")
        else:
            print(s, t, w, "(ok)")

def print_graphs(gs):
    for g in gs:
        print("----------------------------------------------------------------------------")
        print_graph(g)
        print("----------------------------------------------------------------------------")

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))

def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot

def create_feature(features, fields = None):
    return np.hstack(
        [np.array(features[field], dtype=float) for field in fields if field in features] if fields else
        [np.array(features[field], dtype=float) for field in features]
    )
 
def generate_raw_graphs(rand, batch_size, min_max_nodes, geo_density, seet = 0):  
    #------------------ Geometric shape synthetic data ------------------
    num_samples = 1000
    num_points = rand.randint(*min_max_nodes)

    dim_h = int(np.sqrt(num_points))
    dim_w = int(np.sqrt(num_points))
    #num_data = 1000

    #synthetic data
    gen_dataset = GenerateImg(
            dim_x = dim_h, 
            dim_y = dim_w, 
            proportion=(0.05, 0.2, num_samples) 
    )
    gen_dataset.load_data()

    epochs=1
    for epoch in range(epochs):
        gen_trainig = gen_dataset.train_generator( batch_size = batch_size )
        counter = 0
        for gt_graph, set_feature, set_segmentation, in_graph in gen_trainig:
            counter += 1
            nxGraphs = darwin_batches_to_networkx_graphs(gt_graph, set_feature, in_graph, set_segmentation)
            debug(nxGraphs, "raws")
            return nxGraphs

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

#@markdown ##General Visual Params

node_size=200 #@param{type:"slider", min:128, max:2048, step:1}
node_hex_color = "#808080" #@param {type:"string"}
node_color = rgb_from_hex(node_hex_color)
node_linewidth=1.0 #@param{type:"slider", min:0.1, max:3.0, step:0.1}
edge_width=0.2 #@param{type:"slider", min:0.1, max:3.0, step:0.1}
edge_style = "dashed" #@param ["solid", "dashed", "dotted", "dashdot"]
start_color="w"
end_color="k"

#@markdown ##Solution Visual Paramters
solution_node_hex_color = "#3DFF3D" #@param {type:"string"}
solution_node_color = rgb_from_hex(solution_node_hex_color)
solution_node_linewidth=0.6 #@param{type:"slider", min:0.1, max:6.0, step:0.1}
solution_edge_width=6.0 #@param{type:"slider", min:0.1, max:4.0, step=0.1}
solution_edge_style = "solid" #@param ["solid", "dashed", "dotted", "dashdot"]


#@markdown ##Specific Parameters

seed = 5  #@param{type: 'integer'}
rand = np.random.RandomState(seed=seed)

num_examples = 4  #@param{type: 'integer'}


min_nodes = 34 #@param {type:"slider", min:4, max:64, step:1}
max_nodes = 36 #@param {type:"slider", min:4, max:64, step:1}

theta = 12  #@param{type:"slider", min:4, max:64, step:1}
#@markdown Large values (1000+) make trees. Try 20-60 for good non-trees.

horizontal_length = 20 #@param{type: 'integer'}
graphs_per_column = 1 #@param{type: 'integer'}

min_max_nodes = (min_nodes, max_nodes)

graphs = generate_raw_graphs(
    rand,
    num_examples,
    min_max_nodes,
    theta
)

num = 2*min(num_examples, 16)
w = 2*graphs_per_column
size = horizontal_length/w
h = int(np.ceil(num / w))
fig = plt.figure(num=40, figsize=(w*size, h * size))
fig.clf()
for k, raw in enumerate(graphs):
    aux = nx.get_node_attributes(raw,'rgbxy')
    node_rgbxy = []
    for u in aux:
        node_rgbxy.append(aux[len(node_rgbxy)])
    node_rgbxy = np.array(node_rgbxy) 

    x = np.uint32(node_rgbxy[:,-2])
    y = np.uint32(node_rgbxy[:,-1])
    
    img_w, img_h = max(x)+1, max(y)+1

    img_orig = np.zeros(shape=(img_w, img_h, 3))
    img_orig[x,y] = np.uint8(node_rgbxy[:,:3])

    aux = nx.get_node_attributes(raw,'solution')
    node_solution = []
    for u in aux:
        node_solution.append(aux[len(node_solution)])
    node_solution = np.array(node_solution)
    
    img_pred = np.zeros(shape=(img_w, img_h))
    img_pred[x,y] = np.uint8(node_solution)

    src_ax = fig.add_subplot(h, w, 2*k + 1)
    tgt_ax = fig.add_subplot(h, w, 2*k + 2)

    src_ax.set_title('Img Original', fontsize=9)
    src_ax.imshow( img_orig )

    tgt_ax.set_title('Img Prediction', fontsize=9)
    tgt_ax.imshow( img_pred )

    xx, yy = np.meshgrid(
        np.arange(img_h), 
        np.arange(img_w)
    )
    src_ax.plot(xx,yy, 'o', c='w', markersize=1)
    tgt_ax.plot(xx,yy, 'o', c='r', markersize=1)

    for u,v in raw.edges():
        hi = [raw.node[u]["rgbxy"][3],raw.node[u]["rgbxy"][4]]
        wi = [raw.node[v]["rgbxy"][3],raw.node[v]["rgbxy"][4]]
        if raw[u][v]["weight"] >= epsilon:
            src_ax.plot(
                wi, 
                hi, 
                linewidth=0.7, 
                color='w', 
                linestyle='-',
                alpha=raw[u][v]["weight"], 
                marker='o', 
                markersize=1.0 
            )
        if (raw[u][v]["solution"] >= epsilon):
            tgt_ax.plot(
                wi, 
                hi, 
                linewidth=0.7, 
                color='r', 
                linestyle='-',
                alpha = round( raw[u][v]["solution"], 2 ) 
            )

#@title Source and Target from Raw { form-width: "20%" }
def source_from_raw(raw):
    source = nx.DiGraph()
    # Nodes
    fields = ('rgbxy',)
    for node, feature in raw.nodes(data=True):
        source.add_node(
            node, features=create_feature(feature, fields)
        )
    # Edges
    fields = ('weight',)
    for receiver, sender, feature in raw.edges(data=True):
        source.add_edge(
            sender, receiver, features=create_feature(feature, fields)
        )
    
    source.graph["features"] = np.array([0.0])
    
    debug(source, "source")
    
    return source

def target_from_raw(raw):
    target = nx.DiGraph()
    solution_length = 0
    # Nodes
    fields = ('solution', )
    for node, feature in raw.nodes(data=True):
        target.add_node(
            node, features=to_one_hot(
                create_feature(feature, fields).astype(int), 2
            )[0]
        )
    # Edges
    fields = ('solution',)
    for receiver, sender, feature in raw.edges(data=True):
        target.add_edge(
            sender, receiver, features=to_one_hot(
                create_feature(feature, fields).astype(int), 2
            )[0]
        )
        solution_length += int(feature["solution"])
    
    target.graph["features"] = np.array([solution_length], dtype=float)
    
    debug(target, "target")
    
    return target

#@title Helper functions for setup training { form-width: "20%" }

def generate_networkx_graphs(raw_graphs):
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

    source_graphs = [source_from_raw(raw) for raw in raw_graphs]
    target_graphs = [target_from_raw(raw) for raw in raw_graphs]

    return source_graphs, target_graphs


# pylint: disable=redefined-outer-name
def create_placeholders(raw_graphs):
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
    source_graphs = [source_from_raw(raw) for raw in raw_graphs]

    source_ph = utils_tf.placeholders_from_networkxs(
        source_graphs,
        force_dynamic_num_graphs=True
    )

    target_graphs = [target_from_raw(raw) for raw in raw_graphs]

    target_ph = utils_tf.placeholders_from_networkxs(
        target_graphs,
        force_dynamic_num_graphs=True
    )
    debug({
        "source_graphs[0]": source_graphs[0],
        "source_ph" : source_ph,
        "target_graphs[0]" : target_graphs[0],
        "target_ph" : target_ph
    }, "create_placeholders")
    return source_ph, target_ph


def create_loss_ops(target_op, output_ops):
    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
        tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
        for output_op in output_ops
    ]
    return loss_ops


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]

#@title Model definition { form-width: "20%" }

# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model architectures for the demos."""

import sonnet as snt

NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.


def make_mlp_model():
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
        A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
            snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            snt.LayerNorm()
    ])


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=make_mlp_model,
                node_model_fn=make_mlp_model,
                global_model_fn=make_mlp_model
            )

    def _build(self, inputs):
        return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphNetwork(make_mlp_model, make_mlp_model, make_mlp_model)

    def _build(self, inputs):
        return self._network(inputs)


class EncodeProcessDecode(snt.AbstractModule):
    """Full encode-process-decode model.
    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
        global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
        steps. The input to the Core is the concatenation of the Encoder's output
        and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
        the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
        global attributes (does not compute relations etc.), on each message-passing
        step.

                          Hidden(t)   Hidden(t+1)
                             |            ^
                *---------*  |  *------*  |  *---------*
                |         |  |  |      |  |  |         |
      Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
                |         |---->|      |     |         |
                *---------*     *------*     *---------*
    """

    def __init__(
        self,
        edge_output_size=None,
        node_output_size=None,
        global_output_size=None,
        name="EncodeProcessDecode"
    ):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._encoder = MLPGraphIndependent()
        self._core = MLPGraphNetwork()
        self._decoder = MLPGraphIndependent()
        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn)

    def _build(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops

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

tf.reset_default_graph()
 
rand = np.random.RandomState(seed=SEED)

# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters.
num_training_iterations = 10000
theta = 60  # Large values (1000+) make trees. Try 20-60 for good non-trees.
batch_size_tr = 5
batch_size_ge = 100
# Number of nodes per graph sampled uniformly from this range.
num_nodes_min_max_tr = (16, 33) #(32, 65)
num_nodes_min_max_ge = (32, 65) #(64, 129)

# Data.
# Input and target placeholders.
raw_graphs = generate_raw_graphs(rand, num_examples, num_nodes_min_max_tr, theta)
input_ph, target_ph = create_placeholders(raw_graphs)

# Connect the data to the model.
# Instantiate the model.
model = EncodeProcessDecode(edge_output_size=2, node_output_size=2)
# A list of outputs, one per processing step.
debug({"input_ph" : input_ph})
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

#@title Reset session  { form-width: "10%" }

# This cell resets the Tensorflow session, but keeps the same computational
# graph.

try:
    sess.close()
except NameError:
    pass


saver = snt.get_saver(model)
sess = tf.Session()


#saver.restore(sess, "./tmp/model.ckpt")   


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

losses_ge_permuted = []
corrects_ge_permuted = []
solveds_ge_permuted = []

#@title Helper functions for training { form-width: "30%" }

# pylint: disable=redefined-outer-name
def create_feed_dict(sources, targets,source_ph,target_ph):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
        min_max_nodes: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        geo_density: A `float` threshold parameters for the geographic threshold graph's
            threshold. Default= the number of nodes.
        source_ph: The source graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.

    Returns:
        feed_dict: The feed `dict` of source and target placeholders and data.
    """
    source_graphs = utils_np.networkxs_to_graphs_tuple(sources)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {source_ph: source_graphs, target_ph: target_graphs}
    return feed_dict

def compute_accuracy(target, output, use_nodes=False, use_edges=True):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
        target: A `graphs.GraphsTuple` that contains the target graph.
        output: A `graphs.GraphsTuple` that contains the output graph.
        use_nodes: A `bool` indicator of whether to compute node accuracy or not.
        use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        c = []
        if use_nodes:
            xn = np.argmax(td["nodes"], axis=-1)
            yn = np.argmax(od["nodes"], axis=-1)
            c.append(xn == yn)
        if use_edges:
            xe = np.argmax(td["edges"], axis=-1)
            ye = np.argmax(od["edges"], axis=-1)
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved

#@title Run training  { form-width: "10%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.

PERMUTE_TESTS = True
PERMUTE_TRAIN = True

# How much time between logging and printing the current results.
log_every_seconds = 40

var_names = [
    "iteration number",
    "elapsed seconds",
    "training loss",
    "training fraction mse",
    "training fraction examples solved correctly",
]
var_names += [
    "test/generalization loss",
    "test/generalization mse",
    "test/generalization fraction examples solved correctly"
]

if PERMUTE_TESTS:
    var_names += [
        "test/generalization loss with permutations",
        "test/generalization mse with permutations",
        "test/generalization fraction examples solved correctly with permutations"
    ]

print("\t".join(var_names))

labels = [
    "#",
    "T",
    "Ltr",
    "Ctr",
    "Str",
]
labels += [
    "Lge",
    "Cge",
    "Sge"
]
if PERMUTE_TESTS:
    labels += [
        "Lpe",
        "Cpe",
        "Spe"
    ]

print("\t".join(labels))

debug_tags = {""}

start_time = time.time()
last_log_time = start_time
for iteration in range(last_iteration, num_training_iterations):
    last_iteration = iteration
    #Check if it`s time to repeat the dataset
    if iteration % 25 == 0:
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
    
    raw_graphs = generate_raw_graphs(rand, num_examples, num_nodes_min_max_tr, theta)

    if PERMUTE_TRAIN:
        raw_graphs = [
            nx.relabel_nodes(graph, mapping={i: p for i,p in enumerate(np.random.permutation(len(graph)))}) 
            for graph in raw_graphs
        ]
    
    sources, targets = generate_networkx_graphs(raw_graphs)
    feed_dict = create_feed_dict(sources, targets, input_ph, target_ph)
    train_values = sess.run({
            "step": step_op,
            "target": target_ph,
            "loss": loss_op_tr,
            "outputs": output_ops_tr
        },
        feed_dict=feed_dict
    )
    
    correct_tr, solved_tr = compute_accuracy(
        train_values["target"],
        train_values["outputs"][-1],
        use_edges=True
    )
    losses_tr.append(train_values["loss"])
    corrects_tr.append(correct_tr)
    solveds_tr.append(solved_tr)

    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time
    if iteration % 3 == 0: #elapsed_since_last_log > log_every_seconds:
        save_path = saver.save(sess, "./tmp/model.ckpt")
        last_log_time = the_time
        
        raw_graphs_test = generate_raw_graphs(rand, num_examples, num_nodes_min_max_ge, theta)
        
        #Permute raw_graphs
        if PERMUTE_TESTS:
            raw_graphs_permutation = [
                nx.relabel_nodes(graph, mapping={i: p for i,p in enumerate(np.random.permutation(len(graph)))}) 
                for graph in raw_graphs_test
            ]
        
            sources_permutation, targets_permutation = generate_networkx_graphs(raw_graphs_permutation)
            input_ph_permutation, target_ph_permutation = create_placeholders(raw_graphs_permutation)

            # A list of outputs, one per processing step.
            output_ops_ge_permutation = model(input_ph_permutation, num_processing_steps_ge)

            # Test/generalization loss.
            loss_ops_ge_permutation = create_loss_ops(target_ph_permutation, output_ops_ge_permutation)
            loss_op_ge_permutation = loss_ops_ge_permutation[-1]  # Loss from final processing step.

            # Lets an iterable of TF graphs be output from a session as NP graphs.
            input_ph_permutation, target_ph_permutation = make_all_runnable_in_session(input_ph_permutation, target_ph_permutation)

            feed_dict_permutation = create_feed_dict(sources_permutation, targets_permutation, input_ph_permutation, target_ph_permutation)
            
            test_values_permutation = sess.run(
                {
                    "target_permutation": target_ph_permutation,
                    "loss_permutation": loss_op_ge_permutation,
                    "outputs_permutation": output_ops_ge_permutation
                },
                feed_dict=feed_dict_permutation
            )
            correct_ge_permutation, solved_ge_permutation = compute_accuracy(
                test_values_permutation["target_permutation"],
                test_values_permutation["outputs_permutation"][-1],
                use_edges=True
            )
            losses_ge.append(test_values_permutation["loss_permutation"])
            corrects_ge.append(correct_ge_permutation)
            solveds_ge.append(solved_ge_permutation)
        
        sources_test, targets_test = generate_networkx_graphs(raw_graphs_test)
        input_ph_test, target_ph_test = create_placeholders(raw_graphs_test)

        # A list of outputs, one per processing step.
        output_ops_ge_test = model(input_ph_test, num_processing_steps_ge)

        # Test/generalization loss.
        loss_ops_ge_test = create_loss_ops(target_ph_test, output_ops_ge_test)
        loss_op_ge_test = loss_ops_ge_test[-1]  # Loss from final processing step.

        # Lets an iterable of TF graphs be output from a session as NP graphs.
        input_ph_test, target_ph_test = make_all_runnable_in_session(input_ph_test, target_ph_test)

        feed_dict_test = create_feed_dict(sources_test, targets_test, input_ph_test, target_ph_test)
        
        test_values_test = sess.run(
            {
                "target_test": target_ph_test,
                "loss_test": loss_op_ge_test,
                "outputs_test": output_ops_ge_test
            },
            feed_dict=feed_dict_test
        )
        correct_ge_test, solved_ge_test = compute_accuracy(
            test_values_test["target_test"],
            test_values_test["outputs_test"][-1],
            use_edges=True
        )
        losses_ge.append(test_values_test["loss_test"])
        corrects_ge.append(correct_ge_test)
        solveds_ge.append(solved_ge_test)

        elapsed = time.time() - start_time


        logged_iterations.append(iteration)
        row = [
            iteration, 
            elapsed, 
            train_values["loss"], 
            correct_tr, 
            solved_tr, 
        ]
        row += [
            test_values_test["loss_test"],
            correct_ge_test, 
            solved_ge_test
        ]

        if PERMUTE_TESTS:
            row += [
                test_values_permutation["loss_permutation"],
                correct_ge_permutation, 
                solved_ge_permutation
            ]
            
        print("\t".join([str(e) for e in row]))

#@title Visualize results  { form-width: "30%" }

# This cell visualizes the results of training. You can visualize the
# intermediate results by interrupting execution of the cell above, and running
# this cell. You can then resume training by simply executing the above cell
# again.

def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)


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
max_graphs_to_plot = 8 #@param{type:"slider", min:4, max:8, step:1}
num_steps_to_plot = 4 #@param{type:"slider", min:1, max:8, step:1}
node_size = 120 #@param{type:"slider", min:64, max:2048, step:1}
min_c = 0.3
num_graphs = len(raw_graphs)
targets = utils_np.graphs_tuple_to_data_dicts(test_values["target"])
step_indices = np.floor(
    np.linspace(
        0, num_processing_steps_ge - 1,
        num_steps_to_plot
    )
).astype(int).tolist()

outputs = list(zip(*(
    utils_np.graphs_tuple_to_data_dicts(test_values["outputs"][i])
    for i in step_indices
)))
h = min(num_graphs, max_graphs_to_plot)
w = num_steps_to_plot + 1
fig = plt.figure(101, figsize=(18, h * 3))
fig.clf()
ncs = []
for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
    if j >= h:
        break
    ground_truth = target["nodes"][:, -1]
    # Ground truth.
    iax = j * (1 + num_steps_to_plot) + 1
    ax = fig.add_subplot(h, w, iax)
    plotter = GraphPlotter(ax, graph)
    color = {}
    for i, n in enumerate(plotter.nodes):
        color[n] = np.array(
            [1.0 - ground_truth[i], 0.0, ground_truth[i], 1.0]
        ) * (1.0 - min_c) + min_c
    plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        ax.set_facecolor([0.9] * 3 + [1.0])
    except AttributeError:
        ax.set_axis_bgcolor([0.9] * 3 + [1.0])
    ax.grid(None)
    ax.set_title(
        "Ground truth\nSolution length: {}"
            .format(plotter.solution_length)
    )
    # Prediction.
    for k, outp in enumerate(output):
        iax = j * (1 + num_steps_to_plot) + 2 + k
        ax = fig.add_subplot(h, w, iax)
        plotter = GraphPlotter(ax, graph)
        color = {}
        prob = softmax_prob_last_dim(outp["nodes"])
        for i, n in enumerate(plotter.nodes):
            color[n] = np.array(
                [1.0 - prob[n], 0.0, prob[n], 1.0]
            ) * (1.0 - min_c) + min_c
        plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
        ax.set_title(
            "Model-predicted\nStep {:02d} / {:02d}".format(
                step_indices[k] + 1,
                step_indices[-1] + 1
            )
        )