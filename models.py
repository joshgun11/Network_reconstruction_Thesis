
import torch.nn as nn
import torch.nn.functional as F
import torch
#import tensorflow.compat.v1.keras.backend as K
#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
import keras

class Classification(nn.Module):
    def __init__(self,in_size,n_hidden1,n_hidden2,n_hidden3,out_size,p=0):

        super(Classification,self).__init__()
        self.drop=nn.Dropout(p=p)
        self.linear1=nn.Linear(in_size,n_hidden1)
        
        self.linear2=nn.Linear(n_hidden1,n_hidden2)
       
        self.linear3=nn.Linear(n_hidden2,n_hidden3)
        
        self.linear4=nn.Linear(n_hidden3,out_size)
        
    def forward(self,x):
        x=F.relu(self.linear1(x))
        
        x=F.relu(self.linear2(x))
        #x=self.drop(x)
        x=F.relu(self.linear3(x))
        #x=F.relu(self.linear4(x))
        #x=self.drop(x)
        x=self.linear4(x)
        return x

class REG(nn.Module):
    def __init__(self,in_size,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6,n_hidden7,out_size):
        super(REG,self).__init__()
        self.linear1= nn.Linear(in_size, n_hidden1)
        #self.linear1_bn=nn.BatchNorm1d(n_hidden1)
    
        self.linear2=nn.Linear(n_hidden1, n_hidden2)
        #self.linear2_bn=nn.BatchNorm1d(n_hidden2)
    
        self.linear3=nn.Linear(n_hidden2, n_hidden3)
        #self.linear3_bn=nn.BatchNorm1d(n_hidden3)
        
        self.linear4=nn.Linear(n_hidden3, n_hidden4)
        #self.linear4_bn=nn.BatchNorm1d(n_hidden4)
    
        self.linear5=nn.Linear(n_hidden4, n_hidden5)
        self.linear6=nn.Linear(n_hidden5, n_hidden6)
        
        
        self.linear7=nn.Linear(n_hidden6, n_hidden7)
    
        self.linear8=nn.Linear(n_hidden7, out_size)
        
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.zeros_(self.linear3.bias)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        torch.nn.init.zeros_(self.linear4.bias)
        torch.nn.init.xavier_uniform_(self.linear5.weight)
        torch.nn.init.zeros_(self.linear5.bias)
        torch.nn.init.xavier_uniform_(self.linear6.weight)
        torch.nn.init.zeros_(self.linear6.bias)
        torch.nn.init.xavier_uniform_(self.linear7.weight)
        torch.nn.init.zeros_(self.linear7.bias)
        torch.nn.init.xavier_uniform_(self.linear8.weight)
        torch.nn.init.zeros_(self.linear8.bias)
        
        
        
    def forward(self,y):
        y=self.linear1(y)
        y=F.relu(y)
    
        y=self.linear2(y)
        y=F.relu(y)
        
        y=self.linear3(y)
        y=F.relu(y)
        
        y=self.linear4(y)
        y=F.relu(y)
        
        y=self.linear5(y)
        y=F.relu(y)
        y=self.linear6(y)
        y=F.relu(y)
        y=self.linear7(y)
        y=F.relu(y)
        
        y=self.linear8(y)

        return y



class LSTM():
    def __init__(self) -> None:
        pass
    def LSTM_classification(self,n_steps,node_size,num_classes):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(100,activation = "relu",return_sequences = False,input_shape=(n_steps, (node_size-1)*num_classes)))
        model.add(keras.layers.Dense(32,activation = "relu"))

        model.add(keras.layers.Dense(16, activation = "relu"))
        model.add(keras.layers.Dense(num_classes,activation = 'softmax'))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate = 0.01), 
                  loss=keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
        return model

    def LSTM_regression(self,n_steps,node_size,num_classes):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(100,activation = "relu",return_sequences = False,input_shape=(n_steps, node_size-1)))
        model.add(keras.layers.Dense(32,activation = "relu"))

        model.add(keras.layers.Dense(16, activation = "relu"))
        model.add(keras.layers.Dense(num_classes,activation = 'linear'))
        model.compile(optimizer=keras.optimizers.SGD(learning_rate = 0.01),loss='mse')
        return model

    

