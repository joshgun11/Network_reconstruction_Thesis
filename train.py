from models import Classification,REG
from parsers import KParseArgs
from sklearn.model_selection import train_test_split
from preapre_data import KData
import sys

import numpy as np
import torch
from preapre_data import KData

import os
import time

class KTrain():

    def __init__(self) -> None:
        pass
    
    def train_model_classification(self,n_epochs,train_loader,validation_loader,model,node,args):
        #model=Net(48,100,64,32,2,p = 0.2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_epochs = n_epochs
        print_every = 10
        valid_loss_min = np.Inf
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        total_step = len(train_loader)
        last_loss = 100
        patience = 10
        trigger_times = 0
        for epoch in range(1, n_epochs+1):
            model.train()
            running_loss = 0.0
            # scheduler.step(epoch)
            correct = 0
            total=0
            print(f'Epoch {epoch}\n')
            for batch_idx, (data_, target_) in enumerate(train_loader):
                #data_, target_ = data_.to(device), target_.to(device)# on GPU
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                data_ = data_.view(data_.shape[0], args.num_classes*(data_.shape[1]))
                outputs = model(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                _,pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred==target_).item()
                total += target_.size(0)
                if (batch_idx) % 20 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss/total_step)
            print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
            batch_loss = 0
            total_t=0
            correct_t=0
            with torch.no_grad():
                model.eval()
                for data_t, target_t in (validation_loader):
                    #data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
                    data_t = data_t.view(data_t.shape[0], args.num_classes*(data_t.shape[1]))
                    outputs_t = model(data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _,pred_t = torch.max(outputs_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t).item()
                    total_t += target_t.size(0)
                val_acc.append(100 * correct_t / total_t)
                current_loss = batch_loss / len(validation_loader)
                val_loss.append(current_loss)

                network_learned = batch_loss < valid_loss_min
                print(f'validation loss: {current_loss:.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
                # Saving the best weight 
                if network_learned:
                    valid_loss_min = batch_loss
                    if not os.path.exists('trained_models'):
                        os.mkdir('trained_models')
                        os.mkdir('trained_models/classification')
                        os.mkdir('trained_models/classification/'+str(args.data)[:-7])
                        torch.save(model.state_dict(), 'trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
                    elif not os.path.exists('trained_models/classification'):
                        os.mkdir('trained_models/classification')
                        os.mkdir('trained_models/classification/'+str(args.data)[:-7])
                        torch.save(model.state_dict(), 'trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
                    elif not os.path.exists('trained_models/classification/'+str(args.data)[:-7]):
                        os.mkdir('trained_models/classification/'+str(args.data)[:-7])
                        torch.save(model.state_dict(), 'trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
                    else:
                        torch.save(model.state_dict(), 'trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
    



            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break

            else:
                print('trigger times: 0')
                trigger_times = 0

                last_loss = current_loss
    
        model.load_state_dict(torch.load('trained_models/classification/'+str(args.data)[:-7]+'/'+str(node)+'.pt'))
        return model,train_acc,train_loss,val_loss,val_acc


    def train_model_reg(self,n_epochs,train_loader,validation_loader,model,node,args):
        #model=REG(num_input,128,128,128,128,1)
        optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9, weight_decay=0.001)
        criterion = torch.nn.MSELoss()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_epochs = n_epochs
        print_every = 10
        valid_loss_min = np.Inf
        val_loss = []
        train_loss = []
        total_step = len(train_loader)
        last_loss = 100
        patience = 10
        trigger_times = 0
        for epoch in range(1, n_epochs+1):
            model.train()
            running_loss = 0.0
            # scheduler.step(epoch)
            print(f'Epoch {epoch}\n')
            for batch_idx, (data_, target_) in enumerate(train_loader):
                target_ = target_.view(target_.shape[0], 1)
                #data_, target_ = data_.to(device), target_.to(device)# on GPU
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(data_)
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (batch_idx) % 20 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        
            train_loss.append(running_loss/total_step)
            print(f'\ntrain loss: {np.mean(train_loss):.4f}')
            batch_loss = 0
       
        
            with torch.no_grad():
                model.eval()
                for data_t, target_t in (validation_loader):
                    #data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
                    target_t = target_t.view(target_t.shape[0], 1)
                    outputs_t = model(data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
            
                current_loss = batch_loss / len(validation_loader)
                val_loss.append(current_loss)

                network_learned = batch_loss < valid_loss_min
                print(f'validation loss: {current_loss:.4f}\n')
                # Saving the best weight 
                if network_learned:
                    valid_loss_min = batch_loss
                    if not os.path.exists('trained_models'):
                        os.mkdir('trained_models')
                        os.mkdir('trained_models/regression')
                        os.mkdir('trained_models/regression/'+str(args.data)[:-7])
                        torch.save(model.state_dict(), 'trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
                    elif not os.path.exists('trained_models/regression'):
                        os.mkdir('trained_models/regression')
                        os.mkdir('trained_models/regression/'+str(args.data)[:-7])
                        torch.save(model.state_dict(), 'trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
                    elif not os.path.exists('trained_models/regression/'+str(args.data)[:-7]):
                        os.mkdir('trained_models/regression/'+str(args.data)[:-7])
                        torch.save(model.state_dict(), 'trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
                    else:
                        torch.save(model.state_dict(), 'trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt')
                        print('Detected network improvement, saving current model')
            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break

            else:
                print('trigger times: 0')
                trigger_times = 0

                last_loss = current_loss

        model.load_state_dict(torch.load('trained_models/regression/'+str(args.data)[:-7]+'/'+str(node)+'.pt'))
        return model,train_loss,val_loss



    def train(self,args,x,y,node):


        if args.problem_type =="classification":

            st = time.time()
            X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 43,shuffle = False)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
          
            train_test_loader = KData()
            train_loader,validation_loader=train_test_loader.convert_to_tensor(X_train,X_test,y_train,y_test,args.batch_size,args)
            model = Classification(X_train.shape[1]*args.num_classes,100,64,32,args.num_classes,p = 0.2)
            model,train_acc,train_loss,val_loss,val_acc = \
                self.train_model_classification(args.epochs,train_loader,validation_loader,model,node,args)
            et = time.time()
            elapsed_time = et - st
            return x,y,X_test,model,train_acc,train_loss,val_loss,val_acc,y_test,elapsed_time

        elif args.problem_type =="regression":
            st = time.time()
            X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 43,shuffle = False)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
           
                   
            train_test_loader = KData()
            train_loader,validation_loader=train_test_loader.convert_to_tensor(X_train,X_test,y_train,y_test,args.batch_size,args)

            model=REG(X_train.shape[1],128,128,128,128,128,64,64,1)
            model,train_loss,val_loss = \
                self.train_model_reg(args.epochs,train_loader,validation_loader,model,node,args)
            et = time.time()
            elapsed_time = et - st
            return x,y,y_test,X_test,model,train_loss,val_loss,elapsed_time


        
 

    

if __name__=="__main__":
    train_model = KTrain()
    parser = KParseArgs()
    data_selector = KData()
    args = parser.parse_args()
    
    args.plot_cluster = True

    flag = len(sys.argv) == 1
    pairs = data_selector.prepare_data(args.data)


    train_model.train(args,pairs,args.node)

        