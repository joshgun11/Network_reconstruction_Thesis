import torch
import shap
import numpy as np

class SHAP():
    def __init__(self) -> None:
        pass

    def shap_analysis_classification(self,model,X_test,num_classes,node):
        e = shap.DeepExplainer(model,torch.from_numpy(X_test).float()[:300].view(300,num_classes*X_test.shape[1]))
        shap_values = e.shap_values(torch.from_numpy(X_test).float()[300:600].view(300,num_classes*X_test.shape[1]))
        shap_values = np.abs(shap_values[0])+np.abs(shap_values[1])
        shap_values = sum(shap_values)
        def odd_count(n):
            arr = []
            for i in range(0,n,num_classes):
                arr.append(i)
            return arr
        nodes = odd_count(num_classes*X_test.shape[1])
        importances = {}
        for i,j in zip(nodes,range(len(nodes))):
            if j<node:
                total = 0
                for val in range(num_classes):
                    total += shap_values[i+val]
                    importances[str(j)] = total/(X_test.shape[1]/num_classes)
            else:
                total = 0
                for val in range(num_classes):
                    total += shap_values[i+val]
                    importances[str(j+1)]=total/(X_test.shape[1]/num_classes)
                
        return importances


    def shap_analysis_regression(self,model,X_test,num_classes,node):
        e = shap.DeepExplainer(model,torch.from_numpy(X_test).float()[:100].view(100,num_classes*X_test.shape[1]))
        shap_values = e.shap_values(torch.from_numpy(X_test).float()[:100].view(100,num_classes*X_test.shape[1]))
        shap_values = np.abs(shap_values)
        shap_values = sum(shap_values)
    
   
        importances = {}
        for i in range(len(shap_values)):
            if i<node:
                importances[str(i)]=shap_values[i]
            else:
                importances[str(i+1)]=shap_values[i]
            
        return importances


    def shap_analysis_classification_lstm(self,model,X_test,X_train,num_classes,node,n_steps):
    
        e = shap.DeepExplainer(model,X_train[:150])
        shap_values = e.shap_values(X_test[:150])
        shap_values = (np.abs(shap_values[0])+np.abs(shap_values[1]))
        shap_values = sum(shap_values)/num_classes
        shap_values = sum(shap_values)/n_steps
        shap_values = list(shap_values)
        def odd_count(n):
            arr = []
            for i in range(0,n,num_classes):
                arr.append(i)
            return arr
        nodes = odd_count(X_test.shape[2])
        importances = {}
        for i,j in zip(nodes,range(len(nodes))):
            if j<node:
                total = 0
                for val in range(num_classes):
                    total += shap_values[i+val]
                    importances[str(j)] = total/(X_test.shape[2]/num_classes)
            else:
                total = 0
                for val in range(2):
                    total += shap_values[i+val]
                    importances[str(j+1)]=total/(X_test.shape[2]/num_classes)
                
        return importances

    def shap_analysis_lstm_regression(self,model,X_test,X_train,num_classes,node):
        e = shap.DeepExplainer(model,X_train[:150])
        shap_values = e.shap_values(X_test[:150])
        shap_values = np.abs(shap_values)
        shap_values = sum(shap_values)
        shap_values = sum(shap_values)
        shap_values = sum(shap_values)/X_test.shape[1]
    
   
        importances = {}
        for i in range(len(shap_values)):
            if i<node:
                importances[str(i)]=shap_values[i]
            else:
                importances[str(i+1)]=shap_values[i]
 
        return importances