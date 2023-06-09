from reconstruct_graph import Reconstruction
import time
from parsers import KParseArgs
import sys
import numpy as np
from plot import Kplot
from metrics import Kmetrics
import matplotlib.pyplot as plt

# For classification
def experiment_classification(data,graph_type,method,model,node_size,data_size,dynamics,file_name,X,direction,classes,plot):
    parser = KParseArgs()
    args = parser.parse_args()
    reconstructor = Reconstruction()
    args.file_name=file_name+'.csv'
    for i in range(len(data)):
        for j in method:
            args.plot = plot
            try:
                args.direction = direction
                x = X[i]
                args.problem_type = 'classification'
                args.num_classes = classes[i]
                args.node_size = node_size[i]
                args.data = data[i]
                args.method = j
                args.model = model
                args.epochs = 100
                args.dynamics = dynamics[i]
                args.graph = graph_type[i]
                args.data_size = data_size[i]
                args.experiment_name = str(str(args.dynamics)+'_'+str(int(args.data_size/1000))+'K'+'_'+str(args.graph)+'_'+str(args.node_size)+'_'+model+'_'+j+str(x))
                start = time.time()
                symmetric_predicted_matrix,train_loss,train_acc,test_loss,test_acc,times,Acc,run_time = reconstructor.reconstruct(args)
                pred_acc = round(np.mean(Acc),2)
                finish = time.time()
                diff = finish - start
                elapsed_time = diff + np.sum(times)
                print('Execution time:', elapsed_time, 'seconds') 
                graph,adj = reconstructor.ground_truth_graph(args,symmetric_predicted_matrix)
                plotter = Kplot()
                path = str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)
                plotter.plot_avg_metrics(train_acc,test_acc,'Acc',args,path+'/'+'avg_acc')
                plt.close()
                plotter.plot_avg_metrics(train_loss,test_loss,'Loss',args,path+'/'+'avg_loss')
                plt.close()
                metrics = Kmetrics()
                acc = metrics.graph_acc(adj,symmetric_predicted_matrix)
                tp,fp,fn = metrics.tp_fp(adj,symmetric_predicted_matrix)
                loss = metrics.graph_dist(adj,symmetric_predicted_matrix)
                dyn = args.dynamics
                datas = [args.experiment_name,args.graph, str(int(args.data_size/1000))+'K', args.node_size,graph.number_of_edges(), dyn,args.method,args.model,acc,tp,fp,fn,   loss,pred_acc,elapsed_time,args.epochs,int(x)]
                metrics.create_results_data(datas,args.file_name)
                print(args.experiment_name +' Finished Successfully')
            except:
                pass
    print('All experiments Finished')


# For regression
def experiment_regression(data,graph_type,method,model,node_size,data_size,r,file_name,direction,classes,plot):
    parser = KParseArgs()
    args = parser.parse_args()
    reconstructor = Reconstruction()
    
    args.file_name = file_name+'.csv'
    for i in range(len(data)):
        for j in method:
            args.plot = plot
            try:
                args.direction = direction
                st = time.time()
                args.problem_type = 'regression'
                args.num_classes = classes[i]
                args.node_size = node_size[i]
                args.data = data[i]
                args.method = j
                args.model = model
                args.epochs = 100
                args.dynamics = 'cml'
                args.graph = graph_type[i]
                args.data_size = data_size[i]
                args.r = r[i]
                args.experiment_name = str('cml')+'_'+str(int(args.data_size/1000))+'K_'+str(args.r)+'_'+str(args.graph)+'_'+str(args.node_size)+'_'+args.model+'_'+j

                start = time.time()
                symmetric_predicted_matrix,train_loss,test_loss,times,mse,run_time = reconstructor.reconstruct(args)
                pred_acc = round(np.mean(mse),2)
                finish = time.time()
                diff = finish - start
                elapsed_time = diff + np.sum(times)
                print('Execution time:', elapsed_time, 'seconds') 
                graph,adj = reconstructor.ground_truth_graph(args,symmetric_predicted_matrix)
                plotter = Kplot()
                path = str('results/'+args.dynamics+'/'+args.graph+'/'+args.experiment_name)
                plotter.plot_avg_metrics(train_loss,test_loss,'Loss',args,path+'/'+'avg_loss')
                plt.close()
                metrics = Kmetrics()
                acc = metrics.graph_acc(adj,symmetric_predicted_matrix)
                tp,fp,fn = metrics.tp_fp(adj,symmetric_predicted_matrix)
                loss = metrics.graph_dist(adj,symmetric_predicted_matrix)
                dyn = args.dynamics+' '+str(args.r)
                datas = [args.experiment_name,args.graph, str(int(args.data_size/1000))+'K', args.node_size,graph.number_of_edges(), dyn,args.method,args.model,acc,tp,fp,fn,loss,pred_acc,elapsed_time,args.epochs]
                metrics.create_results_data(datas,args.file_name)
                print(args.experiment_name +' Finished Successfully')
            except:
                pass
    
    print('All experiments Finished')


if __name__=="__main__":
    flag = len(sys.argv) == 1

    #Example. You can fill the lists based on experiments you want, and run multiple experiments together
    # with the experiment_classification or experiment_regression functions

    data = ['erdos_5_rps_500_5.pickle'] #can take dataset names
    graph_type = ['erdos'] # should take graph names for each data name. if there are 5 data , should be 5 graph here in order.
    method = ['input_change'] # can take three methods if model is MLP, only input_change for LSTM
    model = 'MLP'
    node_size = [5] #takes number of nodes, for each graph in order
    data_size = [500] # takes data size for each data in order
    dynamics = ['rps'] #takes dynamics name for each data in order
    r = [3.5,3.8] #takes coupled parametr for each data in order in CML case
    direction = 'undirected' # define graphs are undirected or directed 
    X = [5] # takes sampling rate for each data in order
    classes = [3] # takes number of states for each dynamics in order
    file_name = 'check_2' # define file name where results will be stored

    experiment_classification(data,graph_type,method,model,node_size,data_size,dynamics,file_name,X,direction,classes,'no')
    #experiment_regression(data,graph_type,method,model,node_size,data_size,r,'check.csv',direction,classes,'no')

    
    
    
    
    
    
    
    
    
    
    