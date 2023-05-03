import torch
import numpy as np

class Partial_Derivatives():
    def __init__(self) -> None:
        pass

    # For regression
    def gradient_based_sensitivity_regression(self,model, X_test, node, num_classes):
        def calculate_gradients(model, X_test, index, num_classes):
            jacobian_matrix = torch.zeros((X_test.shape[1] * num_classes, num_classes), dtype=torch.float32)
            X_test = torch.from_numpy(X_test)
            my_input = X_test[index].clone().detach().requires_grad_()
            preds = model(my_input.float())
            for i in range(num_classes):
                grd = torch.zeros(num_classes,dtype=torch.float32)
                grd[i] = 1
                model.zero_grad()
                preds.backward(gradient=grd, retain_graph=True)
                jacobian_matrix[:, i] = my_input.grad.view(X_test.shape[1] * num_classes).float()
                my_input.grad.zero_()
            return jacobian_matrix.cpu().numpy()
        J = []
        for i in range(X_test.shape[0]):
            grads = calculate_gradients(model, X_test, i, num_classes)
            grads = np.square(grads)
            J.append(grads)
        J = sum(J) / X_test.shape[0]
        J = np.sqrt(J)
        J_mod = {}
        for i in range(X_test.shape[1]):
            if i < node:
                J_mod[str(i)] = J[i][0]
            else:
                J_mod[str(i + 1)] = J[i][0]
        return J_mod

    # Classification
    def gradient_based_sensitivity_classification(self,model, X_test, node, num_classes):
        def calculate_gradients(model, X_test, index, num_classes):
            X = torch.from_numpy(X_test[index]).view(-1).float()
            X.requires_grad_()
            preds = model(X).view(-1, num_classes)

            Jacobian_matrix = torch.zeros(X.shape[0], num_classes).float()
            for i in range(num_classes):
                model.zero_grad()
                preds[0, i].backward(retain_graph=True)
                Jacobian_matrix[:, i] = X.grad.view(-1).float()
                X.grad.zero_()

            return Jacobian_matrix.numpy()

        nodes = list(range(0, X_test.shape[1] * num_classes, num_classes))
        J = np.mean([np.square(calculate_gradients(model, X_test, i, num_classes)) for i in range(X_test.shape[0])], axis=0)
        J = np.sqrt(J).sum(axis=1) / num_classes

        J_mod = {}
        for i, j in enumerate(nodes):
            total = J[j:j+num_classes].sum()
            J_mod[str(i) if i < node else str(i+1)] = total

        return J_mod
