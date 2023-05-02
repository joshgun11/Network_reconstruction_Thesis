import torch

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from preapre_data import KData
class Input_Change():
    def __init__(self) -> None:
        pass
  
    def permute_col(self,X_test, col_idx):
        X_test_permuted = np.copy(X_test)
        permuted_col = np.random.permutation(X_test_permuted[:, col_idx])
        X_test_permuted[:, col_idx] = permuted_col
        return X_test_permuted

    def feature_importance_class(self,x,y,X_test,num_classes,model,node):
        features = x.shape[1]
    
        with torch.no_grad():
            model.eval()
            X_test = torch.from_numpy(X_test)
            data_t = X_test.view(X_test.shape[0],num_classes*X_test.shape[1])
            predictions = model(data_t.float())
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        importances = {}

        for var in range(features):
            X_test_permuted = self.permute_col(X_test, var)
            with torch.no_grad():
                model.eval()
                new_x_test = torch.from_numpy(X_test_permuted)
                new_x_test = new_x_test.view(new_x_test.shape[0], num_classes*new_x_test.shape[1])
                var_predictions = model(new_x_test.float())
        
            var_predictions = torch.nn.functional.softmax(var_predictions, dim=1)
            s_p = np.abs(predictions-var_predictions)
            s_p = s_p.sum()
            s_p = s_p/X_test.shape[0]
            if node>var:
                importances[str(var)]=s_p
            else:
                importances[str(var+1)]=s_p

        return importances
    def feature_importance_reg(self,x,y,y_test,X_test,model,node):
        features = x.shape[1]
        with torch.no_grad():
            model.eval()
            X_test = torch.from_numpy(np.array(X_test))
            predictions = model(X_test.float())
        importances = {}
   
        for var in range(features):
            X_test_permuted = self.permute_col(X_test, var)
            with torch.no_grad():
                model.eval()
                new_x_test = torch.from_numpy(X_test_permuted)
                var_predictions = model(new_x_test.float())
            diff = mean_absolute_error(predictions,var_predictions)
           
            if var<node:
                importances[str(var)]=diff
            else:
                importances[str(var+1)]=diff

        return importances 


    def permute_col_lstm(self,column,X):
        column = np.random.permutation(X.T[column])
        return column

    def feature_importance_reg_lstm(self,x,y,y_test,X_test,model,node,n_steps):
        prepare_data = KData()
        features = x.shape[2]
    
        predictions = model.predict(X_test)
    
        importances = {}
    
        for var in range(features):
            x_new = x.copy()
            for i in range(x_new.shape[0]):
                x_new[i].T[var] = self.permute_col_lstm(var,x_new[i])
            seq,tar = prepare_data.create_sequence_reg(x_new,y,n_steps)
        
        
            X_train,X_test,y_train,y_test = train_test_split(seq,tar,test_size = 0.2,shuffle = False,random_state = 43)
        
        
            var_predictions = model.predict(X_test)


        
            
            diff = mean_absolute_error(predictions,var_predictions)
       
            if var<node:
                importances[str(var)]=diff
            else:
                importances[str(var+1)]=diff
            
      
           
            
        return importances 


    def feature_importance_class_lstm(self,x,y,X_test,model,node,n_steps,num_classes):
        prepare_data = KData()
        features = int(X_test.shape[2]/num_classes)

        predictions = model.predict(X_test)
        importances = {}
     #variance = np.var(predictions)
        #r_Acc = r2_score(y_test, predictions)
        #Acc = model.evaluate(X_test,y_test)
        for var in range(features):
            data = []
            for i in range(len(x)):
                new = x[i].copy()
            
                new[:,var] =  np.random.permutation(new[:,var])
                data.append(new)
            seq,tar = prepare_data.create_sequence_class(data,y,n_steps)
        
        
        
            X_train,X_test,y_train,y_test = train_test_split(seq,tar,test_size = 0.2,shuffle = False,random_state = 43)
       
            X_test = np.array(X_test)
        
            var_predictions = model.predict(X_test)


            s_p = np.abs(predictions-var_predictions)
            s_p = s_p.sum()
            s_p = s_p/X_test.shape[0]

            if node>var:
                importances[str(var)]=s_p
            else:
                importances[str(var+1)]=s_p
        
        return importances 