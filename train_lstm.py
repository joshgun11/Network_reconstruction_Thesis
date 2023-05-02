
from models import LSTM
import torch.nn as nn
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from preapre_data import KData
from models import LSTM
from tensorflow.keras.callbacks import EarlyStopping
import os
import time


class Train_LSTM():

    def __init__(self) -> None:
        pass
    
    def train_lstm_classification(self,X_train,y_train,X_test,y_test,epochs,n_steps,node_size,num_classes,args,node):
        lstm = LSTM()
        model = lstm.LSTM_classification(n_steps,node_size,num_classes)
        callbacks = [EarlyStopping(monitor="val_loss",patience=5,verbose=1,
                                mode="auto",restore_best_weights=True)]
        history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = epochs,
                    batch_size = 64,callbacks = callbacks)
        if not os.path.exists('trained_models_lstm'):
            os.mkdir('trained_models_lstm')
            os.mkdir('trained_models_lstm/classification')
            os.mkdir('trained_models_lstm/classification/'+str(args.data)[:-7])
            
            model.save('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
            
        elif not os.path.exists('trained_models_lstm/classification'):
            os.mkdir('trained_models_lstm/classification')
            os.mkdir('trained_models_lstm/classification/'+str(args.data)[:-7])
            model.save('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
        elif not os.path.exists('trained_models_lstm/classification/'+str(args.data)[:-7]):
            os.mkdir('trained_models_lstm/classification/'+str(args.data)[:-7])
            model.save('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
        else:
            model.save('trained_models_lstm/classification/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
        return model,history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss']
    
    def train_lstm_regression(self,X_train,y_train,X_test,y_test,epochs,n_steps,node_size,num_classes,args,node):
        lstm = LSTM()
        model = lstm.LSTM_regression(n_steps,node_size,num_classes)
        callbacks = [EarlyStopping(monitor="val_loss",patience=5,verbose=1,
                           mode="auto",restore_best_weights=True)]
        history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = epochs,
                        batch_size = 64,callbacks = callbacks)
        if not os.path.exists('trained_models_lstm'):
            os.mkdir('trained_models_lstm')
            os.mkdir('trained_models_lstm/regression')
            os.mkdir('trained_models_lstm/regression/'+str(args.data)[:-7])
            model.save('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
            
        elif not os.path.exists('trained_models_lstm/regression'):
            os.mkdir('trained_models_lstm/regression')
            os.mkdir('trained_models_lstm/regression/'+str(args.data)[:-7])
            model.save('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
        elif not os.path.exists('trained_models_lstm/regression/'+str(args.data)[:-7]):
            os.mkdir('trained_models_lstm/regression/'+str(args.data)[:-7])
            model.save('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
        else:
            model.save('trained_models_lstm/regression/'+str(args.data)[:-7]+'/'+str(node)+'.h5')
        
        
        return model,history.history['loss'],history.history['val_loss']

    def train_lstm(self,args,x,y,node):


        if args.problem_type =="classification":
            X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 43,shuffle = False)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            start = time.time()
            model,train_acc,train_loss,val_loss,val_acc = self.train_lstm_classification(X_train,y_train,X_test,y_test,args.epochs,args.n_steps,args.node_size,args.num_classes,args,node)
            finish = time.time()
            t = finish - start
            return x,y,X_test,X_train,model,train_acc,train_loss,val_loss,val_acc,y_test,t

        elif args.problem_type =="regression":



            X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 43,shuffle = False)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
           
                   
            start = time.time()
            model,train_loss,val_loss = self.train_lstm_regression(X_train,y_train,X_test,y_test,args.epochs,args.n_steps,args.node_size,args.num_classes,args,node)
            finish = time.time()
            t = finish - start
            return x,y,y_test,X_test,X_train,model,train_loss,val_loss,t
            