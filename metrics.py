import numpy as np 
import csv
import os


class Kmetrics():
    def __init__(self) -> None:
        pass

    def tp_fp(self,adj,pred_adj):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(adj.shape[0]):
            for j in range(adj[i].shape[0]):
                if int(adj[i][j])==1 and pred_adj[i][j]==1:
                    tp = tp+1
                elif int(adj[i][j])==1 and pred_adj[i][j]==0:
                    fn = fn+1
                elif int(adj[i][j])==0 and pred_adj[i][j]==0:
                    tn = tn+1
                elif int(adj[i][j])==0 and pred_adj[i][j]==1:
                    fp = fp+1
        TP = tp/(tp+fn)
        FP = fp/(fp+tn)
        FN = fn/(fn+tp)
        return TP,FP,FN

    def graph_acc(self,adj,pred_adj):
        t = 0
    
        for i in range(adj.shape[0]):
            for j in range(adj[i].shape[0]):
                if int(adj[i][j])==pred_adj[i][j]:
                    t = t+1
        acc = float(t/(adj.shape[0]*adj.shape[1]))
        print(acc)
        return acc

    def graph_dist(self,adj,pred_adj):
        dist = adj-pred_adj
        dist = np.abs(dist)
        dist = np.sum(dist)/2
        return dist

    def create_results_data(self,data,name):
        header = ['Experiment Name','Graph', 'Data Size', 'Node Size','Edge Size','Dynamics','Method','Model','Acc','TP','FP','FN','Loss','Prediction Acc/MSE','Run Time','Epochs','X']
        if not os.path.exists(name):
            with open(name, 'a') as f_object:
 
    
    
                writer_object = csv.writer(f_object)
 
                writer_object.writerow(header)
 
                writer_object.writerow(data)
 
                # Close the file object
                f_object.close()
        else:
            with open(name, 'a') as f_object:
                writer_object = csv.writer(f_object)
 
 
                writer_object.writerow(data)
 
                # Close the file object
                f_object.close()
            
