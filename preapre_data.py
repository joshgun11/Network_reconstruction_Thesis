
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle


class KData():

    def __init__(self) -> None:
        pass
    
    # Data preparation for entering to the classification model
    def prepare_data_classification(self,path,target):
        with open('datasets/'+path,'rb') as f:
            ds = pickle.load(f)

        target_node = target
        y = []
        x = []
        for i in range(len(ds)):
            d = np.zeros((ds[i].shape[0]-1, ds[i].shape[1]-1, ds[i].shape[2]))
            c=[]
            sample_target = ds[i][:,target_node][1:]
            for j in range(ds[i].shape[1]):
                if j!=target_node:
                    other = ds[i][:,j][:-1]
                    if j>target:
                        d[:,j-1] = other
                    else:
                        d[:,j] = other
            x.append(d)
            y.append(sample_target)
        x = np.array(x)
        y = np.array(y)
        x = np.concatenate(x)
        y = np.concatenate(y)
  
        return x,y
    
    # Data preparation for entering to the regression model
    def prepare_data_regression(self,path,target):
        with open('datasets/'+path,'rb') as f:
            df_2 = pickle.load(f)
        target_node = target
        y = []
        x = []
        for i in range(df_2.shape[0]):
            c=[]
            sample_target = df_2[i].T[target_node][1:]
            for j in range(df_2[i].T.shape[0]):
                if j!=target_node:
                    other = df_2[i].T[j]
                    c.append(other[:-1])
            x.append(np.array(c).T)
            y.append(sample_target)
        x = np.array(x)
        y = np.array(y)
        x = np.concatenate(x)
        y = np.concatenate(y)
        return x,y

    # Train and Test loader
    def convert_to_tensor(self,X_train,X_test,y_train,y_test,batch_size,args):
    ## train data
        class TrainData(Dataset):
            def __init__(self, X_data, y_data):
                self.X_data = X_data
                self.y_data = y_data
          
            def __getitem__(self, index):
                return self.X_data[index], self.y_data[index]
        
            def __len__ (self):
                return len(self.X_data)
        ## test data    
        class TestData(Dataset):
    
            def __init__(self, X_data,y_data):
                self.X_data = X_data
                self.y_data = y_data
        
            def __getitem__(self, index):
                return self.X_data[index],self.y_data[index]
        
            def __len__ (self):
                return len(self.X_data)
        BATCH_SIZE = batch_size

        if args.problem_type == 'regression':
            train_data = TrainData(torch.FloatTensor(X_train),torch.FloatTensor(y_train))
            test_data = TestData(torch.FloatTensor(X_test),torch.FloatTensor(y_test))
            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
        elif args.problem_type == 'classification':
            train_data = TrainData(torch.FloatTensor(X_train),torch.argmax(torch.FloatTensor(y_train),dim =1))
            test_data = TestData(torch.FloatTensor(X_test),torch.argmax(torch.FloatTensor(y_test),dim =1))
            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

        return train_loader,test_loader

    # For LSTM
    def prepare_data_classification_lstm(self,path,target):
        with open('datasets/'+path,'rb') as f:
            ds = pickle.load(f)

        target_node = target
        y = []
        x = []
        for i in range(len(ds)):
            d = np.zeros((ds[i].shape[0]-1, ds[i].shape[1]-1, ds[i].shape[2]))
            c=[]
            sample_target = ds[i][:,target_node][1:]
            for j in range(ds[i].shape[1]):
                if j!=target_node:
                    other = ds[i][:,j][:-1]
                    if j>target:
                        d[:,j-1] = other
                    else:
                        d[:,j] = other
            x.append(d)
            y.append(sample_target.reshape(sample_target.shape[0],1,sample_target.shape[1]))
        x = np.array(x)
        y = np.array(y)
      
        return np.array(x),np.array(y)
 
 # Regression LSTM
    def prepare_data_regression_lstm(self,path,target):
        with open('datasets/'+path,'rb') as f:
            df_2 = pickle.load(f)

        target_node = target
        y = []
        x = []
        for i in range(df_2.shape[0]):
            c=[]
            sample_target = df_2[i].T[target_node][1:]
            for j in range(df_2[i].T.shape[0]):
                if j!=target_node:
                    other = df_2[i].T[j]
                    c.append(other[:-1])
            x.append(np.array(c).T)
            y.append(sample_target)
        x = np.array(x)
        y = np.array(y)
        return x,y

    def split_sequences(self,sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)): 
            end_ix = i + n_steps
		    # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
		    # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1,-1:]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def create_sequence_class(self,x,y,n_steps):
        seq = []
        tar = []
        for i in range(len(x)):
            dataset = np.hstack((x[i],y[i].reshape(y[i].shape)))
            # convert into input/output
            X_1, y_a = self.split_sequences(dataset, n_steps)
            seq.append(X_1)
            tar.append(y_a)
        seq = np.concatenate(seq)
        seq = seq.reshape(seq.shape[0],seq.shape[1],seq.shape[2]*seq.shape[3])
        tar = np.concatenate(tar)
        tar = tar.reshape(tar.shape[0],tar.shape[1]*tar.shape[2])
    
        return seq,tar

    def create_sequence_reg(self,x,y,n_steps):
        seq = []
        tar = []
        for i in range(x.shape[0]):
            dataset = np.hstack((x[i],y[i].reshape(-1,1)))
            # convert into input/output
            X_1, y_a = self.split_sequences(dataset, n_steps)
            seq.append(X_1)
            tar.append(y_a)
        seq = np.concatenate(seq)
        tar = np.concatenate(tar)
    
        return seq,tar
        

    


    