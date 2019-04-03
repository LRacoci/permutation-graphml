#@title Darwin's Surfaces { form-width: "20%" }
from scipy.spatial import Delaunay
import numpy as np
from numpy import sin, cos, sqrt
import matplotlib.tri as mtri
from sklearn import preprocessing
import mpl_toolkits.mplot3d as plt3d
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.switch_backend('agg')

import time
np.random.seed(seed=int(time.time()))
#np.random.seed(123) #default seed = 7
eps = 1e-7

SURFACE_TYPES = [
    'elliptic_paraboloid',
    'saddle',
    'torus',
    'ellipsoid',
    'elliptic_hyperboloid',
    'another'
]


def feature_standardization(X):
    '''set_feature.shape: (-1,N,dim)'''
    std_feature = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(X.shape[0]):
        std_feature[i] = (X[i]-np.min(X[i]))/(np.max(X[i])-np.min(X[i]))

    return std_feature

class SurfaceNumpyGenerator:
    '''function surface from http://mathworld.wolfram.com/AlgebraicSurface.html'''

    def __init__(self, num_surfaces, num_points):
        self.num_surfaces = num_surfaces
        self.num_points = num_points

    def saddle_func(self, a, b, h, num_points):
        '''hyperbolic parabolid z = y^2/b^2 - x^2/a^2'''
        n = int(sqrt(num_points))
        x = np.linspace(-n/4, n/4, n)
        y = np.linspace(-n/4, n/4, n)
        x, y = np.meshgrid(x, y)
        z = h * ((y**2/b**2) - (x**2/a**2))
        points2D = np.vstack([x.flatten(), y.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def elliptic_paraboloid_func(self, a, b, c, num_points):
        '''x=x^2/a^2 + y^2/b^2 = z/c'''
        n = int(sqrt(num_points))
        x = np.linspace(-n/4, n/4, n)
        y = np.linspace(-n/4, n/4, n)
        x, y = np.meshgrid(x, y)
        z = c * ((x**2/a**2) + (y**2/b**2))
        points2D = np.vstack([x.flatten(), y.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def cone_func(self, h, r, num_points):
        '''x=((h-u)/h)*r*cos(theta), y=((h-u)/h)*r*sin(theta), z=u'''
        n = int(sqrt(num_points))
        #theta = np.linspace(0, 2*np.pi, n)
        #u = np.linspace(0, h, n)
        #theta, u = np.meshgrid(theta, u)
        #x =  ( (h - u)/h ) * r * cos(theta)
        #y =  ( (h - u)/h ) * r * sin(theta)
        #z = u
        u = np.linspace(-np.pi, np.pi, n)
        v = np.linspace(-np.pi, np.pi, n)
        u, v = np.meshgrid(u,v)

        x = ( (h - u)/h )*np.cos(v)
        y = ( (h - u)/h )*np.sin(v)
        z = u
        #points2D = np.vstack([u.flatten(), theta.flatten()]).T
        points2D = np.vstack([x.flatten(), y.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def elliptic_hyperboloid_func(self, a, b, c, num_points):
        '''x^2/a^2 + y^2/b^2 - z^2/c^2 = 1'''
        n = int(sqrt(num_points))
        u=np.linspace(-4.0,4.0,n);
        v=np.linspace(0, 2*np.pi,n);
        u, v = np.meshgrid(u, v)
        x = a * np.cosh(u) * cos(v)
        y = b * np.cosh(u) * sin(v)
        z = c * np.sinh(u)
        points2D = np.vstack([u.flatten(), v.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def torus_func(self, r, R, num_points):
        '''x=(c+a*cos(u))*cos(v), y=(c+a*cos(v))*sin(u), z=a*sin(v)'''
        n = int(sqrt(num_points))
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, 2 * np.pi, n)
        u, v = np.meshgrid(u, v)
        x = (R + r * cos(v)) * np.cos(u)
        y = (R + r * cos(v)) * np.sin(u)
        z = r * np.sin(v)
        points2D = np.vstack([u.flatten(), v.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def cylinder_func(self, r, h, num_points):
        '''x=r*cos(theta), y=r*sin(theta), z=z'''
        n = int(sqrt(num_points))
        z = np.linspace(0, h, n)
        theta = np.linspace(0, 2*np.pi, n)
        theta, z  = np.meshgrid(theta, z)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points2D = np.vstack([z.flatten(), theta.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def ellipsoid_func(self, a, b, c, num_points):
        '''x=a*cos(u)*sin(v), y=b*sin(u)*sin(v), z=c*cos(v)'''
        n = int(sqrt(num_points))
        v = np.linspace(0, np.pi, n)
        u = np.linspace(0, 2*np.pi, n)
        u, v = np.meshgrid(u, v)
        x = a * cos(u) * sin(v)
        y = b * sin(u) * sin(v)
        z = c * cos(v)
        points2D = np.vstack([u.flatten(), v.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def ding_dong_func(self, r, h, num_points):
        ''' as '''
        n = int(sqrt(num_points))
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(-h, 1, n)
        u, v = np.meshgrid(u, v)
        x = r * v * sqrt(1 - v) * cos(u)
        y = r * v * sqrt(1 - v) * sin(u)
        z = r * v
        points2D = np.vstack([u.flatten(), v.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri


    #http://mathworld.wolfram.com/HyperbolicCylinder.html
    #http://mathworld.wolfram.com/Ding-DongSurface.html
    #http://mathworld.wolfram.com/WhitneyUmbrella.html


    def another_func(self, h, num_points):
        '''cos(abs(x)+abs(y))*(abs(x)+abs(y))'''
        n = int(sqrt(num_points))
        #x = np.arange(-n/4., n/4., 0.5)
        #y = np.arange(-n/4., n/4., 0.5)
        x = np.linspace(-10, 10, n)
        y = np.linspace(-10, 10, n)
        x, y = np.meshgrid(x, y)
        #z = cos(abs(x)+abs(y))*(abs(x)+abs(y))
        z = h * sin(sqrt(x**2 + y**2))
        points2D = np.vstack([x.flatten(), y.flatten()]).T
        tri = Delaunay(points2D).simplices
        return x.flatten(), y.flatten(), z.flatten(), tri

    def transform_translate(self, mtrx, tx, ty, tz):
        '''mtrx.shape(3,3), [xs, ys, zs].T'''
        #convert to homogeneous coord
        mtrx = np.concatenate((mtrx,np.ones((1,mtrx.shape[1]))),axis=0)
        #T matrix transform
        T = np.identity(4)
        T[0][3] = tx
        T[1][3] = ty
        T[2][3] = tz
        #return euclidean coord
        return np.dot(T, mtrx)[:3,:]

    def transform_scale(self, mtrx, sx, sy, sz):
        '''mtrx.shape(3,3), [xs, ys, zs].T'''
        #convert to homogeneous coord
        mtrx = np.concatenate((mtrx,np.ones((1,mtrx.shape[1]))),axis=0)
        #T matrix transform
        T = np.identity(4)
        T[0][0] = sx
        T[1][1] = sy
        T[2][2] = sz
        #return euclidean coord
        return np.dot(T, mtrx)[:3,:]

    def transform_rotate(self, mtrx, alpha, beta, gamma):
        '''mtrx.shape(3,3), [xs, ys, zs].T
             alpha, beta, gamma: angles in radias 90°=1.5708, 180°=3.14159, 270°=4.71239, 360°=6.28319
             alpha: rotate axis x, beta: rotate axis y, gamma: rotate axis z'''
        #get means of xs, ys, zs
        mean_tx = np.mean(mtrx[0])
        mean_ty = np.mean(mtrx[1])
        mean_tz = np.mean(mtrx[2])
        #translate to center axis
        mtrx = self.transform_translate(mtrx, -mean_tx, -mean_ty, -mean_tz)
        #convert to homogeneous coord
        mtrx = np.concatenate((mtrx,np.ones((1,mtrx.shape[1]))),axis=0)
        #T matrix transform
        T = np.identity(4)
        T[0][0]=cos(beta)*cos(gamma); T[0][1]=sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma);
        T[0][2]=cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
        T[1][0]=cos(beta)*sin(gamma); T[1][1]=sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
        T[1][2]=cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)
        T[2][0]=-sin(beta); T[2][1]=sin(alpha)*cos(beta)
        T[2][2]=cos(alpha)*cos(beta)
        #euclidean coord
        mtrx = np.dot(T, mtrx)[:3,:]
        #back to original position
        return self.transform_translate(mtrx, mean_tx, mean_ty, mean_tz)

    def transform_reflex(self, mtrx, xy, xz, yz):
        '''mtrx.shape(3,3), [xs, ys, zs].T
        reflex the axis xy, xz, yz, boolean'''
        #convert to homogeneous coord
        mtrx = np.concatenate((mtrx,np.ones((1,mtrx.shape[1]))),axis=0)
        #T matrix transform
        T = np.identity(4)
        if xy:
            T[2][2] = -1.0
        if xz:
            T[1][1] = -1.0
        if yz:
            T[0][0] = -1.0
        #return euclidean coord
        return np.dot(T, mtrx)[:3,:]

    def transformation_shear(self, mtrx, xy, xz, yz, ca=0., cb=0.,cc=0., cd=0., ce=0., cf=0.):
        '''mtrx.shape(3,3), [xs, ys, zs].T
        shear on plane xy(ca, cb), on plane xz(cc,cd), and yz(ce, cf)'''
        #convert to homogeneous coord
        mtrx = np.concatenate((mtrx,np.ones((1,mtrx.shape[1]))),axis=0)
        #T matrix transform
        T = np.identity(4)
        if xy:
            T[0][2] = ca
            T[1][2] = cb
        if xz:
            T[0][1] = cc
            T[2][1] = cd
        if yz:
            T[1][0] = ce
            T[2][0] = cf
        #return euclidean coord
        return np.dot(T, mtrx)[:3,:]

    def generate_func(self, surface_type, num_surfaces, num_perm):
        list_point_features = []
        list_adj = []
        num_perm -= 1 # because the first object dont has permutation
        #list_point_features_perm = []
        #list_adj_perm = []
        #list_point_features = np.zeros((num_surfaces,self.num_points,3))
        #list_adj = np.zeros((num_surfaces,self.num_points,self.num_points))
        print("dataset " + surface_type)

        for i in range(num_surfaces):
            #create surface points
            if surface_type == 'elliptic_paraboloid':
                a, b = np.random.uniform(low=1.0, high=3.0, size=(2))
                c = np.random.uniform(low=3.0, high=7.0, size=(1))
                #print("a, b: ", a, b)
                x,y,z,tri = self.elliptic_paraboloid_func(a=a, b=b, c=c, num_points=self.num_points)
            elif surface_type == 'cone':
                h = np.random.uniform(low=3.0, high=7.0, size=(1))
                r = np.random.uniform(low=1.0, high=4.0, size=(1))
                x,y,z,tri = self.cone_func(h=h, r=r, num_points=self.num_points)
                #a, b = np.random.uniform(low=1.0, high=3.0, size=(2))
                #c = np.random.uniform(low=4.0, high=7.0, size=(1))
                #print("a, b: ", a, b)
                #x,y,z,tri = self.cone_func(a=a, b=b, c=c, num_points=self.num_points)
            elif surface_type == 'saddle':
                #a, b = np.random.random(2)*10+1.0
                a, b = np.random.uniform(low=0.5, high=3.0, size=(2))
                a = -a if (np.random.randint(2) > 0) else a
                b = -b if (np.random.randint(2) > 0) else b
                h = np.random.uniform(low=3.0, high=8.0, size=(1))
                #print("a, b: ", a, b)
                x,y,z,tri = self.saddle_func(a=a, b=b, h=h, num_points=self.num_points)
            elif surface_type == 'torus':
                R = np.random.uniform(low=2.5, high=5.0, size=(1))
                r = np.random.uniform(low=1.0, high=2.0, size=(1))
                x,y,z,tri = self.torus_func(r=r, R=R, num_points=self.num_points)
            elif surface_type == 'cylinder':
                r = np.random.uniform(low=1.0, high=5.0, size=(1))
                h = np.random.uniform(low=1.0, high=8.0, size=(1))
                x,y,z,tri = self.cylinder_func(r=r, h=h, num_points=self.num_points)
            elif surface_type == 'ellipsoid':
                a, b, c = np.random.uniform(low=1.0, high=10.0, size=(3))
                x,y,z,tri = self.ellipsoid_func(a=a, b=b, c=c, num_points=self.num_points)
            elif surface_type == 'elliptic_hyperboloid':
                a, b = np.random.uniform(low=1.0, high=3.0, size=(2))
                c = np.random.uniform(low=2.0, high=4.0, size=(1))
                x,y,z,tri = self.elliptic_hyperboloid_func(a=a, b=b, c=c, num_points=self.num_points)
            elif surface_type == 'ding_dong':
                h = np.random.uniform(low=1.0, high=8.0, size=(1))
                r = np.random.uniform(low=1.0, high=2.0, size=(1))
                x,y,z,tri = self.ding_dong_func(r=r, h=h, num_points=self.num_points)
            else:
                h = np.random.uniform(low=1.0, high=6.0, size=(1))
                x,y,z,tri = self.another_func(h=h,num_points=self.num_points)

            #triang = mtri.Triangulation(x, y)
            x = np.array(x).reshape(1,-1)
            y = np.array(y).reshape(1,-1)
            z = np.array(z).reshape(1,-1)
            mtrx_point = np.concatenate((x,y,z),axis=0)

            #obtain the adjacency matrix from the surface
            adj = np.zeros((x.shape[1],y.shape[1]))
            #for triangle in range(tri.shape[0]):
                #for i in range(tri.shape[1]):
                #  u = tri[triangle][i]
                #  if i+1 >= tri.shape[1]:
                #    v = tri[triangle][0]
                #  else:
                #    v = tri[triangle][i+1]
                #  adj[u][v] = 1.
                #  adj[v][u] = 1.

            for triangle in range(tri.shape[0]):
                u1 = tri[triangle][0]; v1 = tri[triangle][1];
                u2 = tri[triangle][1]; v2 = tri[triangle][2];
                u3 = tri[triangle][2]; v3 = tri[triangle][0];
                adj[u1][v1] = 1.; adj[v1][u1] = 1.
                adj[u2][v2] = 1.; adj[v2][u2] = 1.
                adj[u3][v3] = 1.; adj[v3][u3] = 1.

            T_mtrx=mtrx_point
            #do serveral transformation on surface
            tx, ty, tz = np.random.uniform(low=0.0, high=10.0, size=(3))
            T_mtrx = self.transform_translate(mtrx=mtrx_point, tx=tx,ty=ty, tz=tz)
            sx, sy, sz = np.random.uniform(low=0.0, high=10.0, size=(3))
            T_mtrx = self.transform_scale(mtrx=T_mtrx, sx=sx, sy=sy, sz=sz)
            alpha, beta, gamma = np.random.uniform(low=0.0, high=6.0, size=(3))
            T_mtrx = self.transform_rotate(mtrx=T_mtrx, alpha=alpha, beta=beta, gamma=gamma)
            xy, xz, yz = np.random.randint(2,size=3)
            T_mtrx = self.transform_reflex(mtrx=T_mtrx, xy=xy, xz=xz, yz=yz)
            #xy, xz, yz = np.random.randint(2,size=3)
            #ca, cb, cc, cd, ce, cf = np.random.uniform(low=0.01, high=2.0, size=(6))
            #T_mtrx = self.transformation_shear(mtrx=T_mtrx, xy=xy, xz=xz, yz=yz, ca=ca, cb=cb, \
            #          cc=cc, cd=cd, ce=ce, cf=cf)

            list_point_features.append(T_mtrx.T)
            list_adj.append(adj)

            #list_point_features[i] = T_mtrx.T
            #list_adj[i] = adj

            T_mtrx_perms, adj_perms = self.get_permutations(
                feature = T_mtrx.T,
                adj = adj,
                num_perm = num_perm
            )
            list_point_features += list(T_mtrx_perms)
            list_adj += list(adj_perms)
            
        return list_point_features, list_adj

    def get_permutations(self, feature, adj, num_perm):
        perm_set = []#set()
        #choose permutation without permutation
        while len(perm_set) < num_perm:
            index_perm = np.random.permutation(self.num_points)
            is_repeat = False
            for m in range(len(perm_set)):
                tmp_repeat = np.array_equal(perm_set[m], index_perm)
                is_repeat = is_repeat and tmp_repeat
            if is_repeat == False:
                perm_set.append(index_perm)

        list_point_features_perm = []
        list_adj_perm = []
        for l in range(num_perm):
            index = perm_set[l]
            T_mtrx_tranp = feature#T_mtrx.T

            T_mtrx_perm = np.zeros_like(T_mtrx_tranp)
            for x in range(self.num_points):
                T_mtrx_perm[index[x]] = T_mtrx_tranp[x]

            adj_perm = np.zeros((self.num_points,self.num_points))

            for x in range(self.num_points):
                for y in range(self.num_points):
                    if adj[x][y] == 1. or adj[y][x] == 1.:
                        adj_perm[index[x]][index[y]] = 1.#adj[x][y]
                        adj_perm[index[y]][index[x]] = 1.

            list_point_features_perm.append(T_mtrx_perm)
            list_adj_perm.append(adj_perm)
        return np.array(list_point_features_perm), np.array(list_adj_perm)

    def preprocessing_permutation(self, feature_graphs, graphs):
        for g in range(len(graphs)):
            prob_permu_graph = np.random.random()
            if prob_permu_graph > 0.5: #permute node of the graph
                adj_mtrx = graphs[g].copy()
                feature_graph = feature_graphs[g].copy()
                num_nodes_permu = int(np.random.random() * self.num_points) #number nodes to permute
                while( num_nodes_permu > 0 ):
                    num_nodes_permu -= 1
                    node1 = np.random.randint(self.num_points-1)
                    node2 = np.random.randint(self.num_points-1)
                    if node1 == node2:
                        num_nodes_permu += 1
                        continue
                    #change features
                    f_tmp = feature_graph[node1].copy()
                    feature_graph[node1] = feature_graph[node2].copy()
                    feature_graph[node2] = f_tmp
                    #change nodes on adj mtrx row
                    adj_tmp = adj_mtrx[node1,:].copy()
                    adj_mtrx[node1,:] = adj_mtrx[node2,:].copy()
                    adj_mtrx[node2,:] = adj_tmp
                    #change nodes on adj mtrx column
                    adj_tmp = adj_mtrx[:,node1].copy()
                    adj_mtrx[:,node1] = adj_mtrx[:,node2].copy()
                    adj_mtrx[:,node2] = adj_tmp
                graphs[g] = adj_mtrx
                feature_graphs[g] = feature_graph
        print("---------------------------")
        return feature_graphs, graphs






def test_SurfaceNumpyGenerator():
    num_surfaces = 18
    num_points = 400
    num_perm = 3
    csurf = SurfaceNumpyGenerator(num_surfaces,num_points)
    types=['elliptic_paraboloid','saddle','torus','ellipsoid','elliptic_hyperboloid','another']
    list_point, list_adj = csurf.generate_func(surface_type='torus',num_surfaces=num_surfaces, num_perm=num_perm)
    print("len(list_point): ", len(list_point))
    print("len(list_adj): ", len(list_adj))

    list_point = np.array(list_point)
    list_point = feature_standardization(list_point)

    print("list_point[0].shape: ", list_point[0].shape)
    print("list_adj[0].shape: ", list_adj[0].shape)
    for i in range(len(list_point)):#(num_surfaces):
        draw_surface(name='surf/surf'+str(i), points_coord=list_point[i], adj=list_adj[i])

#test_SurfaceNumpyGenerator()

class GenerateDataGraphSurface:
    
    def __init__(self, 
        type_dataset='saddle', 
        num_surfaces=100, 
        num_points=100, 
        proportion=(0.8, 0.2),
        proportion_edge=[8./10, 2./10], 
        type_Adj='empty'
    ):
        '''proportion: for training and testing stage
        proportion_edge: proportion of edges and no edges in the adj mtrx'''
        self.csurf = SurfaceNumpyGenerator(num_surfaces,num_points)
        num_perm = 3
        if type_dataset == 'all':
            types=['elliptic_paraboloid','saddle','torus','ellipsoid','elliptic_hyperboloid','another']
            subset = int(num_surfaces/len(types))
            feature_graphs=[]
            graphs=[]
            list_point_train=[]
            list_adj_train=[]
            list_point_test=[]
            list_adj_test=[]
            for t in range(len(types)):
                print("types: ", types[t])
                list_point_t, list_adj_t= self.csurf.generate_func(surface_type=types[t],num_surfaces=subset, num_perm=num_perm)
                list_point_train.extend(list_point_t[0:int(proportion[0]*subset)])
                list_adj_train.extend(list_adj_t[0:int(proportion[0]*subset)])
                list_point_test.extend(list_point_t[int(proportion[0]*subset):])
                list_adj_test.extend(list_adj_t[int(proportion[0]*subset):])

            merge_point_and_adj_train = list(zip(list_point_train, list_adj_train))
            np.random.shuffle(merge_point_and_adj_train)
            feature_graphs_train, graphs_train = zip(*merge_point_and_adj_train)

            merge_point_and_adj_test = list(zip(list_point_test, list_adj_test))
            np.random.shuffle(merge_point_and_adj_test)
            feature_graphs_test, graphs_test = zip(*merge_point_and_adj_test)

            feature_graphs.extend(feature_graphs_train); 
            feature_graphs.extend(feature_graphs_test);
            graphs.extend(graphs_train); graphs.extend(graphs_test);
        else:
            list_point, list_adj = self.csurf.generate_func(surface_type=type_dataset,num_surfaces=num_surfaces, num_perm=num_perm)
            merge_point_and_adj = list(zip(list_point, list_adj))
            np.random.shuffle(merge_point_and_adj)
            feature_graphs, graphs = map(np.array, zip(*merge_point_and_adj))
            #feature_graphs = list_point; graphs = list_adj;

        #feature_graphs = np.array(feature_graphs)
        #graphs = np.array(graphs)
        #------ feature_graphs, graphs = self.csurf.preprocessing_permutation(feature_graphs, graphs)
        #print("----------------------------> ", feature_graphs[0].shape, feature_graphs[1].shape)
        X_features = feature_graphs.reshape(-1,feature_graphs.shape[-1])
        std_scale = preprocessing.StandardScaler().fit(X_features)
        X_std = std_scale.transform(X_features)
        feature_graphs = X_std.reshape(feature_graphs.shape[0],feature_graphs.shape[1],feature_graphs.shape[2])

        #graph_args = create_graphs.Graph_Args(surface_type=type_dataset)
        #graphs_create = create_graphs.create(graph_args)
        #np.random.shuffle(graphs_create)
        self.num_graphs = len(list_point)#num_surfaces
        self.num_nodes = feature_graphs[0].shape[0]
        self.num_features = feature_graphs[0].shape[-1]
        #self.num_edges = graphs_create[0].number_of_edges()

        input_graphs = self.generate_input_graphs(type_Adj, self.num_graphs, self.num_nodes,\
            proportion=proportion_edge)#aleatory

        target_array = self.generate_target(graph_type=type_dataset)

        #self.graphs_test = graphs[int(proportion[0] * self.num_graphs):] #0.8
        #save_graph_list(self.graphs_test, 'gt.dat')
        '''
        graphs_test = graphs[int(proportion[0] * self.num_graphs):] #0.8
        graphs_train = graphs[0:int(proportion[0]*self.num_graphs)] #0.8
        graphs_validate = graphs[0:int(proportion[1]*self.num_graphs)] #0.2

        #print("feature_test: ", surface_type(feature_graphs) )
        feature_test = feature_graphs[int(proportion[0] * self.num_graphs):] #0.8
        #feature_test = np.array(feature_test)
        #shape_feature_test = feature_test.shape
        #noise = np.random.normal(0,7,shape_feature_test[0]*shape_feature_test[1]*shape_feature_test[2])
        #noise = noise.reshape(shape_feature_test[0],shape_feature_test[1],shape_feature_test[2])
        #feature_test += noise
        feature_train = feature_graphs[0:int(proportion[0] * self.num_graphs)] #0.8
        feature_validate = feature_graphs[0:int(proportion[1] * self.num_graphs)] #0.2

        input_graph_test = input_graphs[int(proportion[0] * self.num_graphs):] #0.8
        input_graph_train = input_graphs[0:int(proportion[0] * self.num_graphs)] #0.8
        input_graph_validate = input_graphs[0:int(proportion[1] * self.num_graphs)] #0.2
        '''

        n_training = 0.8 #0.96
        n_eval = 0.1 #0.02
        n_test = 0.1 #0.02
        
        graphs_test = graphs[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        graphs_train = graphs[0:int(self.num_graphs*n_training)] #0.8
        graphs_validate = graphs[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        feature_test = feature_graphs[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        feature_train = feature_graphs[0:int(self.num_graphs*n_training)] #0.8
        feature_validate = feature_graphs[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        input_graph_test = input_graphs[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        input_graph_train = input_graphs[0:int(self.num_graphs*n_training)] #0.8
        input_graph_validate = input_graphs[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        #self.pos_test = pos[int(self.num_graphs*n_training)+int(self.num_graphs*n_eval):] #0.2
        #self.pos_train = pos[0:int(self.num_graphs*n_training)] #0.8
        #self.pos_validate = pos[int(self.num_graphs*n_training):int(self.num_graphs*n_training)+int(self.num_graphs*n_eval)] #0.2

        self.num_val = len(graphs_validate)
        self.num_test = len(graphs_test)
        self.num_training = len(graphs_train)

        self.train_generator = self.batch_generator(graphs_train, feature_train, input_graph_train)
        self.valid_generator = self.batch_generator(graphs_validate, feature_validate, input_graph_validate)
        self.test_generator = self.batch_generator(graphs_test, feature_test, input_graph_test)

        self.target = target_array

        print("DATASET:", type_dataset)
        print("num_graphs:", self.num_graphs)
        print("num_nodes by graph:", self.num_nodes)
        #print("num_edges by graph:", self.num_edges)
        print("num_features by node:", self.num_features)
        print("num_training:", self.num_training)
        print("num_val:", self.num_val)
        print("num_test:", self.num_test)

    def generate_input_graphs( self, surface_type, num_graphs, num_nodes, proportion=[8./10, 2./10] ):
        '''proportion: 8/10 no edges and 2/10 edges'''
        inputs_graphs = []
        for i in range(num_graphs):
            if surface_type == 'empty':
                graph_i = np.identity(num_nodes)#np.zeros((num_nodes,num_nodes))
            elif surface_type == 'full':
                graph_i = np.ones((num_nodes,num_nodes))
            else: #'aleatory' #p=propostion of 0's and 1's
                graph_i = np.random.choice([0, 1], size=(num_nodes,num_nodes), p=proportion)
                graph_i = graph_i * graph_i.T
            np.fill_diagonal(graph_i, 1.0)
            inputs_graphs.append(graph_i.astype(np.float32))
        return inputs_graphs

    def generate_target( self, graph_type):
        #types=['elliptic_paraboloid','saddle','torus','ellipsoid','elliptic_hyperboloid','another']
        target_array = [0,0,0,0,0,0]
        if graph_type == 'elliptic_paraboloid':
            target_array[0] = 1
        elif graph_type == 'saddle':
            target_array[1] = 1
        elif graph_type == 'torus':
            target_array[2] = 1
        elif graph_type == 'ellipsoid':
            target_array[3] = 1
        elif graph_type == 'elliptic_hyperboloid':
            target_array[4] = 1
        else:
            target_array[5] = 1
        return target_array

    def batch_generator( self, db_graph, db_feature, db_input_graph ):
        def gen_batch( batch_size ):
            for offset in range(0, len(db_graph), batch_size):
                files_graph = db_graph[offset:offset+batch_size]
                files_feature = db_feature[offset:offset+batch_size]
                files_input_graph = db_input_graph[offset:offset+batch_size]

                yield np.array( files_graph ), np.array( files_feature ), np.array( files_input_graph )
        return gen_batch



def save_graph_csv(name, points_coord, adj, dim):
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

def save_graph_merge(name, points_coord, adj_pred, adj_gt, dim):
    '''points_coord.shape: (num_points, coord=3),
        adj_pred, adj_gt.shape:(num_points,num_points)'''

    x = points_coord[:,0]
    y = points_coord[:,1]
    if dim == 3:
        z = points_coord[:,2]

    f_merge = open(name+'_merge.csv', 'w')
    for i in range(adj_pred.shape[0]):
        for j in range(i,adj_pred.shape[1]):
            if i != j and ( adj_pred[i][j]==1 or adj_gt[i][j]==1):
                f_merge.write(str(x[i])+", "+str(y[i])),
                if dim == 3:
                    f_merge.write(", "+str(z[i])),
                f_merge.write(", "+str(x[j])+", "+str(y[j])),
                if dim == 3:
                    f_merge.write(", "+str(z[j])),

                if adj_pred[i][j] > adj_gt[i][j]: #blue
                    f_merge.write(", c"),
                elif adj_pred[i][j] < adj_gt[i][j]: #red
                    f_merge.write(", b"),
                else: #if adj_pred[i][j]==1 and adj_gt[i][j]==1: #black
                    f_merge.write(", a"),
                f_merge.write("\n")
    f_merge.close()

def save_H(name, features):
    '''features.shape: (num_points, dim)'''
    np.savetxt(name+'_feature.csv', features, delimiter=",", fmt='%10.6f')