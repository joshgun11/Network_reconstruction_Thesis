
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from preapre_data import KData
from train import KTrain
from parsers import KParseArgs
import keras
import sys
import os
import torch
from plot import Kplot
from graphs import KGraph
from Senisitivity_Analysis_Methods.Input_change import Input_Change
from Senisitivity_Analysis_Methods.partial_derivatives import Partial_Derivatives
from Senisitivity_Analysis_Methods.shap_analysis import SHAP
from Senisitivity_Analysis_Methods.Deeplift import Deeplift
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import time
import pickle
from models import Classification,REG

########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

class Sensitiviy_Analysis():

    def __init__(self) -> None:
        pass

    def prepare_scores(self,scores_dict,args):
        results = []
        for i in scores_dict.values():
            results.append(i)
        results = np.array(results)
        if args.problem_type == "regression":
            if args.method =="gradient_based":
                results = results.reshape(args.node_size-1)
        return results
########################################################################################################################################################################################
#CLustering method
    def clustering(self,args,results,X,node):
        scores = sorted(results)
        scores = np.array(scores)
        cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
        cluster.fit(scores.reshape(-1, 1))
        labels = cluster.labels_
        labels = list(cluster.labels_)
        num_clust1 = list(cluster.labels_).count(labels[0])
        thresh = scores[num_clust1-1]+0.000000000001
    
        labels = (results> thresh).astype('float')
        return labels,thresh
########################################################################################################################################################################################
# SA methods
    def apply_method(self,args,node):
        print('Process is going for node: '+str(node))
        train_model = KTrain() 
        ploter = Kplot()
        graph = KGraph()
        prepare_data = KData()
        ######################################
        if args.problem_type == 'classification':
            if args.model == 'LSTM':
                from train_lstm import Train_LSTM
                lstm_trainer = Train_LSTM()
                x,y = prepare_data.prepare_data_classification_lstm(args.dynamics+'/'+args.data,node)
                x_seq,y_seq = prepare_data.create_sequence_class(x,y,args.n_steps)
                if not os.path.exists('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'.h5'):
                    x_seq,y_seq,X_test,X_train,model,train_acc,train_loss,val_loss,val_acc,y_test,t = lstm_trainer.train_lstm(args,x_seq,y_seq,node)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','wb') as f:
                        pickle.dump(train_loss, f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','wb') as f:
                        pickle.dump(val_loss, f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_acc.pickle','wb') as f:
                        pickle.dump(train_acc, f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_acc.pickle','wb') as f:
                        pickle.dump(val_acc, f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','wb') as f:
                        pickle.dump(t, f)
                    t = 0
                else:
                    X_train,X_test,y_train,y_test = train_test_split(x_seq,y_seq,test_size = .2,random_state = 43,shuffle = False)
                    model = keras.models.load_model('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','rb') as f:
                        train_loss = pickle.load(f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_acc.pickle','rb') as f:
                        train_acc = pickle.load(f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','rb') as f:
                        val_loss = pickle.load(f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_acc.pickle','rb') as f:
                        val_acc = pickle.load(f)
                    with open('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','rb') as f:
                        t = pickle.load(f)
        ######################################
            else:
                x,y = prepare_data.prepare_data_classification(args.dynamics+'/'+args.data,node)
                if not os.path.exists('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt'):
                    x,y,X_test,model,train_acc,train_loss,val_loss,val_acc,y_test,t = train_model.train(args,x,y,node)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','wb') as f:
                        pickle.dump(train_loss, f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','wb') as f:
                        pickle.dump(val_loss, f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_acc.pickle','wb') as f:
                        pickle.dump(train_acc, f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_acc.pickle','wb') as f:
                        pickle.dump(val_acc, f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','wb') as f:
                        pickle.dump(t, f)
                    t = 0
                else:
                    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 43,shuffle = False)
                    model = Classification(X_train.shape[1]*args.num_classes,100,64,32,args.num_classes,p = 0.2)
                    model.load_state_dict(torch.load('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt'))
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','rb') as f:
                        train_loss = pickle.load(f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_acc.pickle','rb') as f:
                        train_acc = pickle.load(f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','rb') as f:
                        val_loss = pickle.load(f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_val_acc.pickle','rb') as f:
                        val_acc = pickle.load(f)
                    with open('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','rb') as f:
                        t = pickle.load(f)
        ######################################
        elif args.problem_type == 'regression':
            if args.model == 'LSTM':
                from train_lstm import Train_LSTM
                lstm_trainer = Train_LSTM()
                x,y = prepare_data.prepare_data_regression_lstm(args.dynamics+'/'+args.data,node)
                x_seq,y_seq = prepare_data.create_sequence_reg(x,y,args.n_steps)
                if not os.path.exists('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'.h5'):
                    x_seq,x_seq,y_test,X_test,X_train,model,train_loss,val_loss,t = lstm_trainer.train_lstm(args,x_seq,y_seq,node)
                    with open('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','wb') as f:
                        pickle.dump(train_loss, f)
                    with open('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','wb') as f:
                        pickle.dump(val_loss, f)
                    with open('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','wb') as f:
                        pickle.dump(t, f)
                    t = 0
                else:
                    X_train,X_test,y_train,y_test = train_test_split(x_seq,y_seq,test_size = .2,random_state = 43,shuffle = False)
                    model = keras.models.load_model('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
                    with open('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','rb') as f:
                        train_loss = pickle.load(f)
                    
                    with open('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','rb') as f:
                        val_loss = pickle.load(f)
                    
                    with open('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','rb') as f:
                        t = pickle.load(f)
        ######################################
            elif args.model == 'MLP':
                x,y = prepare_data.prepare_data_regression(args.dynamics+'/'+args.data,node)
                if not os.path.exists('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt'):
                    x,y,y_test,X_test,model,train_loss,val_loss,t = train_model.train(args,x,y,node)
                    with open('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','wb') as f:
                        pickle.dump(train_loss, f)
                    with open('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','wb') as f:
                        pickle.dump(val_loss, f)
                    
                    with open('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','wb') as f:
                        pickle.dump(t, f)
                    t = 0
                else:
                    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 43,shuffle = False)
                    model = REG(X_train.shape[1],128,128,128,128,128,64,64,1)
                    model.load_state_dict(torch.load('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt'))
                    with open('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_loss.pickle','rb') as f:
                        train_loss = pickle.load(f)
                    
                    with open('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'_val_loss.pickle','rb') as f:
                        val_loss = pickle.load(f)
                    
                    with open('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'_train_time.pickle','rb') as f:
                        t = pickle.load(f)
        ######################################

########################################################################################################################################################################################
# Input Permutation
        if args.method =="input_change":
            analysis  = Input_Change()
            if args.problem_type == 'classification':
                if args.model =="MLP":
                    #x,y,X_test,model,train_acc,train_loss,val_loss,val_acc,y_test = train_model.train(args,x,y)
                    scores_dict = analysis.feature_importance_class(x,y,X_test,args.num_classes,model,node)
                    with torch.no_grad():
                        model.eval()
                        X_test = torch.from_numpy(X_test)
                        data_t = X_test.view(X_test.shape[0],args.num_classes*X_test.shape[1])
                        predictions = model(data_t.float())
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                    predictions = torch.argmax(predictions,dim = 1)
                    y_test = torch.from_numpy(y_test)
                    y_test = torch.argmax(y_test,dim = 1)
                    acc = accuracy_score(y_test,predictions)

                    
                elif args.model == 'LSTM':
                    scores_dict = analysis.feature_importance_class_lstm(x,y,X_test,model,node,args.n_steps,args.num_classes)
                    
                    predictions = model.predict(X_test)
                    predictions = np.argmax(predictions,axis = 1)
                    
                    y_test = np.argmax(y_test,axis = 1)
                    acc = accuracy_score(y_test,predictions)


            elif args.problem_type =='regression':
                if args.model =="MLP": 
                
                    scores_dict = analysis.feature_importance_reg(x,y,y_test,X_test,model,node)
                    print(scores_dict)
                    with torch.no_grad():
                        model.eval()
                        X_test = torch.from_numpy(np.array(X_test))
                        predictions = model(X_test.float())
                    y_test = torch.from_numpy(y_test)
                    mse = mean_squared_error(y_test,predictions)
                    
                elif args.model == "LSTM":
                    
                    scores_dict = analysis.feature_importance_reg_lstm(x,y,y_test,X_test,model,node,args.n_steps)
                    predictions = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test,predictions)

########################################################################################################################################################################################
#Partial Derivatives
        elif args.method =="gradient_based":
            analysis = Partial_Derivatives()
            if args.problem_type =='regression':
                scores_dict = analysis.gradient_based_sensitivity_regression(model,X_test,node,1)
                print(scores_dict)
                with torch.no_grad():
                    model.eval()
                    X_test = torch.from_numpy(np.array(X_test))
                    predictions = model(X_test.float())
                y_test = torch.from_numpy(y_test)
                mse = mean_squared_error(y_test,predictions)
           
            elif args.problem_type == 'classification':
                scores_dict = analysis.gradient_based_sensitivity_classification(model,X_test,node,args.num_classes)
                with torch.no_grad():
                    model.eval()
                    X_test = torch.from_numpy(X_test)
                    data_t = X_test.view(X_test.shape[0],args.num_classes*X_test.shape[1])
                    predictions = model(data_t.float())
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                predictions = torch.argmax(predictions,dim = 1)
                y_test = torch.from_numpy(y_test)
                y_test = torch.argmax(y_test,dim = 1)
                acc = accuracy_score(y_test,predictions)

########################################################################################################################################################################################

# Additinal method SHAP: Not used in thesis
        elif args.method =="shap":

            analysis = SHAP()
            if args.problem_type =='regression':
                if args.model =="MLP":
                    scores_dict = analysis.shap_analysis_regression(model,X_test,1,node)
                    with torch.no_grad():
                        model.eval()
                        X_test = torch.from_numpy(np.array(X_test))
                        predictions = model(X_test.float())
                    y_test = torch.from_numpy(y_test)
                    mse = mean_squared_error(y_test,predictions)

                elif args.model == 'LSTM':
                    scores_dict = analysis.shap_analysis_lstm_regression(model,X_test,X_train,args.num_classes,args.node)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test,predictions)

            elif args.problem_type == 'classification':

                if args.model =="MLP":
                    scores_dict = analysis.shap_analysis_classification(model,X_test,args.num_classes,node)
                    with torch.no_grad():
                        model.eval()
                        X_test = torch.from_numpy(X_test)
                        data_t = X_test.view(X_test.shape[0],args.num_classes*X_test.shape[1])
                        predictions = model(data_t.float())
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                    predictions = torch.argmax(predictions,dim = 1)
                    y_test = torch.from_numpy(y_test)
                    y_test = torch.argmax(y_test,dim = 1)
                    acc = accuracy_score(y_test,predictions)

                elif args.model == 'LSTM':
                    scores_dict = analysis.shap_analysis_classification_lstm(model,X_test,X_train,args.num_classes,args.node,args.n_steps)
                    predictions = model.predict(X_test)
                    predictions = np.argmax(predictions,axis = 1)
                    y_test = np.argmax(y_test,axis = 1)
                    acc = accuracy_score(y_test,predictions)

        #results = self.prepare_scores(scores_dict,args)       
        #labels = self.clustering(args,results,scores_dict,node = node)
 ########################################################################################################################################################################################
# DeepLift
        elif args.method =="deeplift":
            analysis = Deeplift()
            if args.problem_type =='regression':
                if args.model =="MLP":
                    scores_dict = analysis.DeepLift_Imp(model,X_test,1,node,np.mean(X_test))
                    with torch.no_grad():
                        model.eval()
                        X_test = torch.from_numpy(np.array(X_test))
                        predictions = model(X_test.float())
                    y_test = torch.from_numpy(y_test)
                    mse = mean_squared_error(y_test,predictions)

            elif args.problem_type == 'classification':
                if args.model =="MLP":
                    scores_dict = analysis.DeepLift_Imp(model,X_test,args.num_classes,node,0)
                    with torch.no_grad():
                        model.eval()
                        X_test = torch.from_numpy(X_test)
                        data_t = X_test.view(X_test.shape[0],args.num_classes*X_test.shape[1])
                        predictions = model(data_t.float())
                    predictions = torch.nn.functional.softmax(predictions, dim=1)
                    predictions = torch.argmax(predictions,dim = 1)
                    y_test = torch.from_numpy(y_test)
                    y_test = torch.argmax(y_test,dim = 1)
                    acc = accuracy_score(y_test,predictions)
                
        results = self.prepare_scores(scores_dict,args)       
        labels,thresh = self.clustering(args,results,scores_dict,node = node)  

        if args.plot =='yes':
            st = time.time()
            ploter.plot_all(args,graph,node,scores_dict,results,labels,ploter)
            et = time.time()
            plotting_time =(et - st)
            plotting_time = t - plotting_time
        else:
            plotting_time = t
            
        print(plotting_time)


########################################################################################################################################################################################
        if args.problem_type=="classification":
            return labels,results,train_loss,train_acc,val_loss,val_acc,plotting_time,acc,thresh
        elif args.problem_type =="regression":
            return labels,results,train_loss,val_loss,plotting_time,mse,thresh
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

if __name__=="__main__":

    # You can apply a SA to a single node as in example by args.node argument:

    method = Sensitiviy_Analysis()
    parser = KParseArgs()
    prepare_Data = KData()
    args = parser.parse_args()
    args.node_size = 10
    args.data = 'erdos_10_cml_3.55_5000.pickle'
    args.graph = 'erdos'
    args.dynamics = 'cml'
    args.node = 0
    args.method = 'gradient_based'
    args.model = 'MLP'
    args.problem_type = 'regression'
    args.num_classes  = 1
    args.data_size = 5000
    args.experiment_name = 'a'
    args.epochs = 10
    args.r = 3.55
    args.plot_cluster = True

    flag = len(sys.argv) == 1
    if args.problem_type =="classification":
        labels,results,train_loss,train_acc,val_loss,val_acc,plotting_time,acc = method.apply_method(args,args.node)
    elif args.problem_type =="regression":
        labels,results,train_loss,val_loss,plotting_time,mse = method.apply_method(args,args.node)
    
    print(labels)
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################









    

    