from cmath import sqrt
import networkx as nx
import numpy as np
import math

class KGraph():

    def __init__(self) -> None:
        pass

    def generate_graph(self,args,node_size):
        NUM_NODES = node_size
        np.random.seed(0)
        # grid will be generated n x n always. If you set number of nodes 10, it will be 3x3 graph and 9 nodes. 
        if args.graph=='grid':
            alpha = int(math.sqrt(NUM_NODES))
            G =  nx.grid_2d_graph(alpha,alpha)
            g = nx.convert_node_labels_to_integers(G)
            return g

        elif args.graph=='geom':
            G_geom_small = nx.random_geometric_graph(NUM_NODES, 0.3, seed=43)
            geom = nx.convert_node_labels_to_integers(G_geom_small)
            return geom

        elif args.graph=='albert':
            albert = nx.barabasi_albert_graph(NUM_NODES, 2, seed=0)
            return albert
            
        # Can generate directed or undirected erdos graph based on direction argument
        elif args.graph=='erdos':
            if args.direction == 'directed':
                G_erdos_small = nx.erdos_renyi_graph(NUM_NODES, 0.15, seed=43,directed = True)
            else:
                G_erdos_small = nx.erdos_renyi_graph(NUM_NODES, 0.15, seed=43)
            erdos = nx.convert_node_labels_to_integers(G_erdos_small)
            return erdos

