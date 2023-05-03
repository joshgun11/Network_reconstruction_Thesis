from multiprocessing.spawn import prepare
from sensitivity_analysis import Sensitiviy_Analysis
from parsers import KParseArgs
import numpy as np
import networkx as nx
import sys
from graphs import KGraph
from preapre_data import KData
import pandas as pd
import matplotlib.pyplot as plt
import os
from plot import Kplot
from metrics import Kmetrics
import time



class Reconstruction():

    def __init__(self) -> None:
        pass
    
    #Apply model training and SA to each node and return unsymmetric binary matrix, and weight matrix
    def construct_graph(self,args):
        method = Sensitiviy_Analysis()
        nodes = range(args.node_size)
        predicted_matrix = []

        scores = []
        Train_Acc = []
        Train_Loss = []
        Test_Acc = []
        Test_Loss  =[]
        times = []
        Acc = []
        MSE = []
        thresholds = []

        for node in nodes:
            if args.problem_type == 'classification':
                predicted_labels,results,train_loss,train_acc,test_loss,test_acc,plotting_time,acc,thresh = method.apply_method(args,int(node))
                Train_Acc.append(train_acc)
                Train_Loss.append(train_loss)
                Test_Acc.append(test_acc)
                Test_Loss.append(test_loss)
                times.append(plotting_time)
                Acc.append(acc)
                thresholds.append(thresh)
            else:
                predicted_labels,results,train_loss,test_loss,plotting_time,mse,thresh= method.apply_method(args,int(node))
                Train_Loss.append(train_loss)
                Test_Loss.append(test_loss)
                times.append(plotting_time)
                MSE.append(mse)
                thresholds.append(thresh)

            scores.append(results)
            predicted_labels = list(predicted_labels)
            predicted_labels.insert(int(node), 0)
            predicted_matrix.append(predicted_labels)

        if args.problem_type == 'classification':
            return np.array(predicted_matrix),np.array(scores),Train_Acc,Train_Loss,Test_Loss,Test_Acc,times,Acc,thresholds
        else:
            return np.array(predicted_matrix),np.array(scores),Train_Loss,Test_Loss,times,MSE,thresholds
 
    # Symmetrize binary adj matrix based on weights (scores)
    def symmetrize(self,args,predicted_matrix,scores,thresholds):
        new_scores = []
        for i in range(args.node_size):
            new_scores.append(np.insert(scores[i],i,1.0))
        for i in range(predicted_matrix.shape[0]):
            for j in range(predicted_matrix[i].shape[0]):
                if predicted_matrix[i][j]!=predicted_matrix[j][i]:
                    mean_score = (new_scores[i][j]+new_scores[j][i])/2
                    if mean_score>=(thresholds[i]+thresholds[j])/2:
                        predicted_matrix[i,j]=1
                        predicted_matrix[j,i]=1
                    else:
                        predicted_matrix[i,j]=0
                        predicted_matrix[j,i]=0
        return predicted_matrix

    # Reconstruct binary, symmetric adj matrix and save
    def reconstruct(self,args):
        start = time.time()
        if args.problem_type == 'classification':
            predicted_matrix,scores,train_acc,train_loss,test_loss,test_acc,times,Acc,thresholds = self.construct_graph(args)
        else:
            predicted_matrix,scores,train_loss,test_loss,times,MSE,thresholds= self.construct_graph(args)
        
        if args.direction =='directed':
            symmetric_predicted_matrix = predicted_matrix.T
        else:
            symmetric_predicted_matrix = self.symmetrize(args,predicted_matrix,scores,thresholds)

        finish = time.time()
        run_time = finish - start

        if not os.path.exists('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name):
            os.makedirs('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name, exist_ok=True)
            with open(str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)+'/predicted_adj_matrix.npy', 'wb') as f:
                 np.save(f, symmetric_predicted_matrix)
            with open(str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)+'/scores.npy', 'wb') as f:
                 np.save(f, scores)
        else:
            with open(str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)+'/predicted_adj_matrix.npy', 'wb') as f:
                 np.save(f, symmetric_predicted_matrix)
            with open(str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)+'/scores.npy', 'wb') as f:
                 np.save(f, scores)

        if args.problem_type == 'classification':
            return symmetric_predicted_matrix,train_loss,train_acc,test_loss,test_acc,times,Acc,run_time
        else:
            return symmetric_predicted_matrix,train_loss,test_loss,times,MSE,run_time

    def ground_truth_graph(self,args,pred_matrix):
        graph_generator = KGraph()
        pred_graph = nx.from_numpy_matrix(pred_matrix)
        pred_graph = nx.convert_node_labels_to_integers(pred_graph)

     #Generate ground truth graph for plotting (It is not used in method)
        graph =  graph_generator.generate_graph(args,args.node_size)
        graph = nx.convert_node_labels_to_integers(graph)
        org_adj_matrix = nx.to_numpy_array(graph)
        if not os.path.exists('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name):
            os.makedirs('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name, exist_ok=True)
            with open(str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)+'/original_adj_matrix.npy', 'wb') as f:
                np.save(f, org_adj_matrix)
        else:
            with open(str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)+'/original_adj_matrix.npy', 'wb') as f:
                np.save(f, org_adj_matrix)

        edges = graph.edges()
        edges_pred = pred_graph.edges()

        # Plot groudn truth and reconstructed graph
        plt.figure(figsize=(20,10))
        colors = ['green' if (u,v) in edges else 'lightcoral' for u,v in edges_pred]
        ax = plt.gca()
        ax.set_title('Reconstructed Graph-'+'-Node size:'+str(args.node_size)+'--Graph: '+str(args.graph) )
        nx.draw(pred_graph, with_labels=True, pos=nx.kamada_kawai_layout(graph),ax = ax,node_color = 'grey',font_color = 'black', edge_color = colors)
        plot_path = str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)
        plt.savefig(plot_path+'/'+str(args.method)+'_'+str(args.model)+'_'+str(args.graph)+'_'+str(args.node_size)+'_'+str(args.data_size)+'_reconstructed.png')
        plt.close()
        plt.figure(figsize=(20,10))
        ax = plt.gca()
        ax.set_title('Ground Truth Graph-'+'-Node size:'+str(args.node_size)+'--Graph: '+str(args.graph) )
        nx.draw(graph, with_labels=True, pos=nx.kamada_kawai_layout(graph),ax = ax,node_color = 'grey',font_color = 'black', edge_color = 'green')
        plot_path = str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)
        plt.savefig(plot_path+'/'+str(args.method)+'_'+str(args.model)+'_'+str(args.graph)+'_'+str(args.node_size)+'_'+str(args.data_size)+'_original.png')
        plt.close()
        
        return graph,org_adj_matrix


if __name__=="__main__":
    import time
    st = time.time()
    parser = KParseArgs()
    args = parser.parse_args()
    reconstructor = Reconstruction()
    
    if args.problem_type == 'classification':
        symmetric_predicted_matrix,train_loss,train_acc,test_loss,test_acc,times,Acc,run_time = reconstructor.reconstruct(args)
        pred_acc = np.mean(Acc)
    else:
        symmetric_predicted_matrix,train_loss,test_loss,times,MSE,run_time = reconstructor.reconstruct(args)
        pred_acc = np.mean(test_loss)
    et = time.time()
    elapsed_time = et - st + np.sum(times)
    print('Execution time:', elapsed_time, 'seconds') 

    graph,adj = reconstructor.ground_truth_graph(args,symmetric_predicted_matrix)

    plotter = Kplot()
    path = str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)
    if args.problem_type == 'classification':
        plotter.plot_avg_metrics(train_acc,test_acc,'Acc',args,path+'/'+'avg_acc')
        plt.close()

    plotter.plot_avg_metrics(train_loss,test_loss,'Loss',args,path+'/'+'avg_loss')
    plt.close()

    metrics = Kmetrics()
    acc = metrics.graph_acc(adj,symmetric_predicted_matrix)
    tp,fp,fn = metrics.tp_fp(adj,symmetric_predicted_matrix)
    loss = metrics.graph_dist(adj,symmetric_predicted_matrix)
    if args.problem_type == 'regression':
        dyn = args.dynamics+' '+str(args.r)
    else:
        dyn = args.dynamics

    datas = [args.experiment_name,args.graph, str(int(args.data_size/1000))+'K', args.node_size,graph.number_of_edges(), dyn,args.method,args.model,acc,tp,fp,fn,loss,pred_acc,elapsed_time,args.epochs]
    metrics.create_results_data(datas,args.file_name+'.csv')
    print(args.experiment_name +' Finished Successfully')
    







        
