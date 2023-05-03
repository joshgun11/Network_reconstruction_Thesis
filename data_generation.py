
import numpy as np
import networkx as nx

from graphs import KGraph
from parsers import KParseArgs
import sys
import os
import random
import pickle
class Generate_Data():

    def __init__(self) -> None:
        pass

    
    # Voter
    def voter(self,args):

        def gen_voter_trajectory(G, noise=0.01, burn_in=100, length=1000, every_x_step=1):
            A = [1.,0]
            B = [0.,1.]
            states = [random.choice([A, B]) for i in range(G.number_of_nodes())]
            step = -1
            trajectory = list()
            while True:
                step += 1
                rates = np.zeros(G.number_of_nodes())
                for n in range(G.number_of_nodes()):
                    rates[n] = len([n_j for n_j in G.neighbors(n) if states[n_j] != states[n]]) + noise
                    rates[n] = 1.0/rates[n] # numpy uses mean as rate param
        
                jump_time = np.random.exponential(rates)
        
                change_n = np.argmin(jump_time)
        
                states[change_n] = A if states[change_n] == B else B    
                if step > burn_in and step % every_x_step == 0:
                    trajectory.append(list(states))
                if len(trajectory) == length:
                    break
            return trajectory

        def generate_voter_data(G,n,every_x_step):
            data = []
            for i in range(n):
                sim = gen_voter_trajectory(G, noise=0.01, burn_in=10, length=1000, every_x_step=every_x_step)
                data.append(sim)
                if np.concatenate(data).shape[0]>=n:
                    break
            return np.array(data)

            
        graph_generator = KGraph()
        G = graph_generator.generate_graph(args,args.node_size)
        TS = generate_voter_data(G,args.data_size,args.every_x_step)
        
        plot_path = 'datasets/voter'
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            os.mkdir('datasets/voter')
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_voter_'+ str(args.data_size) + '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        elif not os.path.exists(plot_path) :
            os.mkdir(plot_path)
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_voter_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        else:
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_voter_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)

        return print('Data Generated Successfully')

    # SIS
    def sis(self,args):
        def gen_sis_trajectory(G, inf_rate=1.0, rec_rate=2.0, noise=0.1, burn_in=10, length=1000, every_x_step=5):
            S = [1., 0.]
            I = [0., 1.]
            trajectory = list()
            steps = burn_in + length
            states = [random.choice([S, I]) for i in range(G.number_of_nodes())]
            step = -1
            while True:
                step += 1
                rates = np.zeros(G.number_of_nodes())
                for n in range(G.number_of_nodes()):
                    rates[n] = noise
                    if states[n] == I:
                        rates[n] += rec_rate
                    if states[n] == S:
                        rates[n] += inf_rate * len([n_j for n_j in G.neighbors(n) if states[n_j] == I])
                    rates[n] = 1.0/rates[n] # numpy uses mean as rate param
                jump_time = np.random.exponential(rates)
                change_n = np.argmin(jump_time)
                states[change_n] = S if states[change_n] == I else I     
                if step > burn_in and step % every_x_step == 0:
                    trajectory.append(np.array(states))
                if len(trajectory) == length:
                    break
            return np.array(trajectory)
            
        def generate_sis_data(G,n,every_x_step):
            data = []
            for i in range(n):
                sim = gen_sis_trajectory(G, inf_rate=args.beta, rec_rate=args.mu, noise=0.1, burn_in=10, length=1000, every_x_step=every_x_step)
                data.append(np.array(sim))
                if np.concatenate(data).shape[0]>=n:
                    break
            return np.array(data)

        graph_generator = KGraph()
        G = graph_generator.generate_graph(args,args.node_size)
        TS = generate_sis_data(G,args.data_size,args.every_x_step)

        plot_path = 'datasets/sis'
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            os.mkdir('datasets/sis')
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_sis_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        elif not os.path.exists(plot_path) :
            os.mkdir(plot_path)
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_sis_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        else:
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_sis_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)

        return print('Data Generated Successfully')

    #Game of Life
    def gol(self,args):
        def gen_gameoflife_trajectory(G, change_rate=1.0, noise=0.01, burn_in=100, length=1000, every_x_step=1):
            A = [1., 0.]
            D = [0., 1.]
            step = -1
            states = [random.choice([A, D]) for i in range(G.number_of_nodes())]
            trajectory = list()
            while True:
                step += 1
                rates = np.zeros(G.number_of_nodes())
                for n in range(G.number_of_nodes()):
                    alive_neighbors =  len([n_j for n_j in G.neighbors(n) if states[n_j] == A])
                    dead_neighbors =  len([n_j for n_j in G.neighbors(n) if states[n_j] == D])
                    alive_frac = alive_neighbors/(alive_neighbors+dead_neighbors)
                    rates[n] = noise
                    if states[n] == A:
                        rates[n] += (alive_neighbors+dead_neighbors) - np.abs(alive_neighbors-dead_neighbors)
                    if states[n] == D:
                        rates[n] +=  np.abs(alive_neighbors-dead_neighbors)
                    rates[n] = 1.0/rates[n] # numpy uses mean as rate param
                jump_time = np.random.exponential(rates)
                change_n = np.argmin(jump_time)
                states[change_n] = A if states[change_n] == D else D    
                if step > burn_in and step % every_x_step == 0:
                    trajectory.append(list(states))
                if len(trajectory) == length:
                    break
            return trajectory
        def generate_gol_data(G,n,every_x_step):
            data = []
            for i in range(n):
                sim = gen_gameoflife_trajectory(G, change_rate=1.0, noise=0.01, burn_in=10, length=1000, every_x_step=every_x_step)
                data.append(np.array(sim))
                if np.concatenate(data).shape[0]>=n:
                    break
            return np.array(data)
        
        graph_generator = KGraph()
        G = graph_generator.generate_graph(args,args.node_size)
        TS = generate_gol_data(G,args.data_size,args.every_x_step)
        
        plot_path = 'datasets/gol'
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            os.mkdir('datasets/gol')
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_gol_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        elif not os.path.exists(plot_path) :
            os.mkdir(plot_path)
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_gol_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        else:
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_gol_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)

        return print('Data Generated Successfully')

    # CML
    def cml(self,args):
        def solve_cmp(G, s=args.s, r=args.r,limit = True):
            def f_map(x): return r*x*(1.0-x)
            states = [np.array([random.random() for j in range(G.number_of_nodes())])]
            steps = 100 + random.choice(range(10))

            for _ in range(steps):
                candidate = states[_].copy()
                new_states = list(candidate)
                for n in G.nodes():
                    v = candidate[n]
                    neig = list(G.neighbors(n))
                    new_states[n] = (1.0-s) * f_map(v) + s/len(neig) * np.sum([f_map(candidate[n_j]) for n_j in neig])
                candidate = list(new_states)
                states.append(candidate)
            if limit:
               states = states[:20]
        
        
            return states

        def gen_cmp(G, s=args.s, r=args.r):
          
            states = solve_cmp(G, s, r,limit = True)
    
            return np.array(states)
        
        def gen_data_cml(G,N):
            data = []
            n = int(N/20)
            for i in range(n):
                sample = gen_cmp(G, s=args.s, r=args.r)
                data.append(sample)
                
            return np.array(data)

        graph_generator = KGraph()
        G = graph_generator.generate_graph(args,args.node_size)
        TS = gen_data_cml(G,args.data_size)
        
        plot_path = 'datasets/cml'
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            os.mkdir('datasets/cml')
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_cml_'+ str(args.r)+'_'+str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        elif not os.path.exists(plot_path) :
            os.mkdir(plot_path)
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_cml_'+ str(args.r)+'_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        else:
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_cml_'+ str(args.r)+'_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)

        return print('Data Generated Successfully')

    # RPS
    def rps(self,args):
        def gen_rps(G, change_rate=1.0, noise=0.1,burn_in=10, length=1000, every_x_step=5):
            R = [1, 0, 0]
            P = [0, 1, 0]
            S = [0, 0, 1]
            step = -1
            states = [random.choice([R,P,S]) for i in range(G.number_of_nodes())]
            trajectory = list()
            while True:
                step += 1
                rates = np.zeros(G.number_of_nodes())
                for n in range(G.number_of_nodes()):
                    rates[n] = noise
                    if states[n] == R:
                        rates[n] += change_rate * len([n_j for n_j in G.neighbors(n) if states[n_j] == P]) # paper wins against rock
                    if states[n] == P:
                        rates[n] += change_rate * len([n_j for n_j in G.neighbors(n) if states[n_j] == S])
                    if states[n] == S:
                        rates[n] += change_rate * len([n_j for n_j in G.neighbors(n) if states[n_j] == R])
                rates[n] = 1.0/rates[n] # numpy uses mean as rate param
                jump_time = np.random.exponential(rates)
                change_n = np.argmin(jump_time)
                if states[change_n] == R:
                    states[change_n] = P
                elif states[change_n] == P:
                    states[change_n] = S
                elif states[change_n] == S:
                    states[change_n] = R
            
                if step > burn_in and step % every_x_step == 0:
                    trajectory.append(list(states))
                if len(trajectory) == length:
                    break
        
            return trajectory
        def generate_rps_data(G,n,every_x_step):
            data = []
            for i in range(n):
                sim = gen_rps(G, change_rate=1.0, noise=0.1,burn_in=10, length=1000, every_x_step=every_x_step)
                data.append(np.array(sim))
                if np.concatenate(data).shape[0]>=n:
                    break
            return np.array(data)
            
        graph_generator = KGraph()
        G = graph_generator.generate_graph(args,args.node_size)
        TS = generate_rps_data(G,args.data_size,args.every_x_step)
        
        plot_path = 'datasets/rps'
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            os.mkdir('datasets/rps')
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_rps_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        elif not os.path.exists(plot_path) :
            os.mkdir(plot_path)
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_rps_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        else:
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_rps_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)

        return print('Data Generated Successfully')
    
    # Forest Fire
    def forest_fire(self,args):
        def gen_forestfire(G, growth_rate=1.0, lightning_rate=0.1, firespread_rate = 2.0,
            fireextinct_rate = 2.0, noise=0.01,burn_in=10, length=1000, every_x_step=5):
            E = [1, 0, 0] # empty
            T = [0, 1, 0] # tree
            L = [0, 0, 1] # fire
            step = -1
            states = [random.choice([E,T,L]) for i in range(G.number_of_nodes())]
            trajectory = list()
            while True:
                step += 1
                rates = np.zeros(G.number_of_nodes())
      
                for n in range(G.number_of_nodes()):
                    rates[n] = noise
                    if states[n] == E:
                        rates[n] += growth_rate
                    if states[n] == T:
                        rates[n] += firespread_rate if len([n_j for n_j in G.neighbors(n) if states[n_j]==L])>0 else 0.0
                        rates[n] += lightning_rate
                    if states[n] == L:
                        rates[n] += fireextinct_rate
                    rates[n] = 1.0/rates[n] # numpy uses mean as rate param
                jump_time = np.random.exponential(rates)
                change_n = np.argmin(jump_time)
                if states[change_n] == E:
                    states[change_n] = T
                elif states[change_n] == T:
                    states[change_n] = L
                elif states[change_n] == L:
                    states[change_n] = E
                if step > burn_in and step % every_x_step == 0:
                    trajectory.append(list(states))
                if len(trajectory) == length:
                    break
            
        
            return trajectory
        def generate_forest_data(G,n,every_x_step):
            data = []
            for i in range(n):
                sim = gen_forestfire(G, growth_rate=1.0, lightning_rate=0.1, firespread_rate = 2.0,
            fireextinct_rate = 2.0, noise=0.01,burn_in=10, length=1000, every_x_step=every_x_step)
                data.append(np.array(sim))
                if np.concatenate(data).shape[0]>=n:
                    break
            return np.array(data)

        graph_generator = KGraph()
        G = graph_generator.generate_graph(args,args.node_size)
        TS = generate_forest_data(G,args.data_size,args.every_x_step)
        
        plot_path = 'datasets/forest_fire'
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
            os.mkdir('datasets/forest_fire')
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_forest_fire_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        elif not os.path.exists(plot_path) :
            os.mkdir(plot_path)
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_forest_fire_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)
        else:
            with open(plot_path+'/'+str(args.graph)+'_'+str(args.node_size)+'_forest_fire_'+ str(args.data_size)+ '_'+str(args.every_x_step)+'.pickle','wb') as f:
                pickle.dump(TS, f)

        return print('Data Generated Successfully')




        

if __name__=='__main__':
    parser = KParseArgs()
    args = parser.parse_args()
    flag = len(sys.argv) == 1
    data_generator = Generate_Data()
    if args.dynamics =='voter':
        data_generator.voter(args)
    elif args.dynamics == 'sis':
        data_generator.sis(args)
    elif args.dynamics == 'gol':
        data_generator.gol(args)
    elif args.dynamics == 'cml':
        data_generator.cml(args)
    elif args.dynamics == 'rps':
        data_generator.rps(args)
    elif args.dynamics == 'forest_fire':
        data_generator.forest_fire(args)
#'sis','voter','rps','forest_fire'



    



