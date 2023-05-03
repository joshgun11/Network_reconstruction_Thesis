import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
import pandas as pd 

class Kplot():

    def __init__(self) -> None:
        pass

    def feature_importance_plot(self,G,pair,results,c,args,path = None):
        if args.direction =='directed':
            neighbours = list(G.predecessors(pair))
        else:
            neighbours = list(G.neighbors(pair))
        def plot_color_label_before(results, neighbours):
            neigh = {key:results[key] for key in results.keys() if int(key)  in neighbours}
            non_neigh = {key:results[key] for key in results.keys() if int(key) not in neighbours}
            plt.bar(x=results.keys(), height=c,color = 'white', alpha=0.7)
            plt.bar(x=neigh.keys(), height=neigh.values(), color='green', alpha=0.7,label='Neighbour')
            plt.bar(x=non_neigh.keys(), height=non_neigh.values(), color='red', alpha=0.7,label='Non-Neighbour')
            

            if path:
                plt.legend()
                plt.title('Feature Importance Plot For Node: '+str(pair))
                plt.xlabel('Nodes '+'Info: '+
                str(('Model :'+args.model,'Graph :'+args.graph,'Dynamics :'+args.dynamics,'Method :'+args.method)))
                plt.ylabel('Importance Scores')
            
                plt.savefig(path+'.png')
                plt.close()
                
        
        plot_color_label_before(results, neighbours)

        return print("Plot created succesfully")


    def clustering_plot(self,labels,results,X,node,args,path):
       
        colors = ['green' if node ==1 else 'red' for node in labels]
        plt.scatter(np.array(list(X.keys())),results, c=colors, cmap='rainbow')
        plt.title('Clustering Results For Node: '+str(node))
        plt.xlabel('Nodes '+'Info: '+str(('Model :'+args.model,'Graph :'+args.graph,'Dynamics :'+args.dynamics,'Method :'+args.method)))
        plt.ylabel('Importance Scores')
        plt.savefig(path+'.png')
    
        plt.close()


    def plot_avg_metrics(self,metric_train,metric_test,metric,args,path):
        def find_max_list(list):
            list_len = [len(i) for i in list]
            return max(list_len)
        max_length = find_max_list(metric_train)
        for i in metric_train:
            if len(i)<max_length:
                diff = max_length-len(i)
                i += diff * [i[-1]]
        for i in metric_test:
            if len(i)<max_length:
                diff = max_length-len(i)
                i += diff * [i[-1]]
        metric_train = np.array(metric_train)
        metric_test = np.array(metric_test)
        metric_train = sum(metric_train)/metric_train.shape[0]
        metric_test = sum(metric_test)/metric_test.shape[0]
        plt.plot(metric_train,color = 'green',label = 'Train')
        plt.plot(metric_test,color ='red',label = 'Test')
        plt.legend()
        if metric == 'Acc':
            plt.title('Average Train And Test Accuracy')
            plt.xlabel("Number of Epochs "+'Info: '+str(['Model: '+args.model,'Graph: '+args.graph,"Dynamics: "+args.dynamics]))
            plt.ylabel("Average Accuracy")
        else:
            plt.title('Average Train And Test Loss')
            plt.xlabel("Number of Epochs "+'Info: '+str(['Model: '+args.model,'Graph: '+args.graph,"Dynamics: "+args.dynamics]))
            plt.ylabel("Average Loss")
        plt.savefig(path+'.png')
        plt.close()


    def heatmap(self,g,results,node,args,path):
        results = (results-np.mean(results))/np.std(results)
    
        values = list(results)
        values.insert(node,5)
        color_map = dict(zip(list(g.nodes()), values))
        val_map = color_map
        values = [val_map.get(node) for node in g.nodes()]
        ax = plt.gca()
        ax.set_title('Heatmap based on importance score for Node '+str(node) )
        nx.draw(g, with_labels=True,node_size=1000,ax = ax, node_color=values, cmap=plt.cm.Reds,pos=nx.kamada_kawai_layout(g))
        plt.savefig(path+'.png')
        plt.close()
    
    def distance_and_importance(self,G,results,X,target,args,path=None):
        distance = []
        for i in G.nodes():
            try:
                dist = nx.shortest_path_length(G, source=i, target=target)
            except:
                dist = 10
            distance.append(dist)
        distance.remove(0)
        neighbours = list(G.neighbors(target))
        colors = ['red' if int(key) in neighbours else 'blue' for key in X.keys()]
        df = pd.DataFrame({'value':results,'dist':distance,'category':colors})
        scatter = plt.scatter(df.dist,df.value,s=50,c=df.category.astype('category').cat.codes)
        plt.legend(handles=scatter.legend_elements()[0], 
            labels=['Non-Neighbour','Neighbour'])
        plt.xlabel('Shortest Distance to the Node '+str(target)) 
        plt.ylabel('Importance Score') 
  
        # displaying the title
        plt.title("Node's distance and its importance to Target Node")
        if path:
        
            plt.savefig(path+'.png')
            plt.close()

    def plot_all(self,args,graph,node,scores_dict,results,labels,ploter):
            data_size = args.data_size
            node_size = args.node_size
            G = graph.generate_graph(args,node_size)
            if args.experiment_name == None:
                args.experiment_name = str(str(args.dynamics)+'_'+str(int(args.data_size/1000))+'K'+'_'+str(args.graph)+'_'+str(args.node_size)+'_'+args.model+'_'+args.method+str(args.every_x_step))

            path = 'results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/'
            if not os.path.exists('results'):
                os.mkdir('results')
                os.mkdir('results/'+str(args.dynamics))
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph))
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name)
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Feature_Importances')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/CLustering')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Heatmaps')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Distance_plot')
                ploter.feature_importance_plot(G,node,scores_dict,results,args,path+'/Feature_Importances/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.clustering_plot(labels,results,scores_dict,node,args,path+'/Clustering/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.heatmap(G,results,node,args,path+'/Heatmaps/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.distance_and_importance(G,results,scores_dict,node,args,path+'/Distance_plot/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()

            elif not os.path.exists('results/'+str(args.dynamics)) :
                os.mkdir('results/'+str(args.dynamics))
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph))
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name)
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Feature_Importances')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/CLustering')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Heatmaps')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Distance_plot')
                ploter.feature_importance_plot(G,node,scores_dict,results,args,path+'/Feature_Importances/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.clustering_plot(labels,results,scores_dict,node,args,path+'/Clustering/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.heatmap(G,results,node,args,path+'/Heatmaps/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.distance_and_importance(G,results,scores_dict,node,args,path+'/Distance_plot/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
            elif not os.path.exists('results/'+str(args.dynamics)+'/'+str(args.graph)) :
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph))
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name)
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Feature_Importances')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/CLustering')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Heatmaps')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Distance_plot')
                ploter.feature_importance_plot(G,node,scores_dict,results,args,path+'/Feature_Importances/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.clustering_plot(labels,results,scores_dict,node,args,path+'/Clustering/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.heatmap(G,results,node,args,path+'/Heatmaps/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.distance_and_importance(G,results,scores_dict,node,args,path+'/Distance_plot/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
            elif not os.path.exists('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name) :
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name)
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Feature_Importances')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/CLustering')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Heatmaps')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Distance_plot')
                ploter.feature_importance_plot(G,node,scores_dict,results,args,path+'/Feature_Importances/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.clustering_plot(labels,results,scores_dict,node,args,path+'/Clustering/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.heatmap(G,results,node,args,path+'/Heatmaps/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.distance_and_importance(G,results,scores_dict,node,args,path+'/Distance_plot/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
            elif not os.path.exists('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Feature_Importances') :
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Feature_Importances')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/CLustering')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Heatmaps')
                os.mkdir('results/'+str(args.dynamics)+'/'+str(args.graph)+'/'+args.experiment_name+'/Distance_plot')
                ploter.feature_importance_plot(G,node,scores_dict,results,args,path+'/Feature_Importances/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.clustering_plot(labels,results,scores_dict,node,args,path+'/Clustering/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.heatmap(G,results,node,args,path+'/Heatmaps/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.distance_and_importance(G,results,scores_dict,node,args,path+'/Distance_plot/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
            else:
        
                ploter.feature_importance_plot(G,node,scores_dict,results,args,path+'/Feature_Importances/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.clustering_plot(labels,results,scores_dict,node,args,path+'/Clustering/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.heatmap(G,results,node,args,path+'/Heatmaps/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()
                ploter.distance_and_importance(G,results,scores_dict,node,args,path+'/Distance_plot/'+str(args.method)+'_'+str(args.model)+'_'+str(node)+'_'+str(node_size)+'_'+str(data_size))
                plt.close()