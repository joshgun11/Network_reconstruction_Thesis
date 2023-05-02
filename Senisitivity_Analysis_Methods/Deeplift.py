from captum.attr import DeepLift
import torch 
import numpy as np 


class Deeplift():
    def __init__(self) -> None:
        pass
    def DeepLift_Imp(self,model,inputs,num_classes,node,baseline):
    
        inputs = np.array(inputs)
        def feature_importances(model,inputs,t):
            """
                Computes feature importances for a PyTorch MLP model using the DeepLift method.

                Parameters:
                    - model (nn.Module): A PyTorch MLP model.
                    - inputs (torch.Tensor): A batch of input data to compute feature importances for.

                Returns:
                    - importances (np.ndarray): An array of feature importances.
                """
            model.eval()
            deep_lift = DeepLift(model,multiply_by_inputs=True)
            # compute the attribution scores for the input
            attribution = deep_lift.attribute(inputs, target=t,baselines = baseline)
            # calculate the sum of the attribution scores for each feature
            #attribution
            #importances = np.sum(attribution, axis=0)
            return attribution
        def odd_count(n):
            arr = []
            for i in range(0,n,num_classes):
                arr.append(i)
            return arr
        nodes = odd_count(num_classes*inputs.shape[1])
        contrubutions = []
        for i in range(num_classes):
            importance = feature_importances(model,torch.from_numpy(inputs).float().view(inputs.shape[0],num_classes*inputs.shape[1]),i)
            importance = importance.detach().numpy()
            importance = np.abs(importance)
            importance = np.sum(importance,axis = 0)/inputs.shape[0]
            imp = []
            if num_classes>1:
                for i in nodes:
                    t = importance[i]+importance[i+1]
                    imp.append(t)
                contrubutions.append(imp)
            else:
                contrubutions.append(importance)
        contrubutions = np.array(contrubutions)
    
        contrubutions = np.sum(contrubutions,axis = 0)
        contrubutions = contrubutions/num_classes
        results = {}
        for i in range(len(contrubutions)):
            if i>=node:
                results[str(i+1)] = contrubutions[i]
            else:
                results[str(i)] = contrubutions[i]
            
        return results