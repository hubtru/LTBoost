import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Trains NLinear than trains LGBM channelwise on residuals
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in 
        self.individual = configs.individual
        
        self.Tree = []
        self.tree_lr = configs.tree_lr
        self.tree_loss = configs.tree_loss
        self.treelb = min(configs.tree_lb, self.seq_len) # Lags for tree (must be less or equal then seq_len)
        self.lb_data = configs.lb_data
        self.num_leaves = configs.num_leaves
        self.tree_iter = configs.tree_iter
        self.psmooth = configs.psmooth
        self.num_jobs = configs.num_jobs

        self.device =  'cuda' if configs.use_gpu else 'cpu'

        self.normalize = configs.normalize
        self.use_RevIN = configs.use_revin
        self.use_sigmoid = configs.use_sigmoid

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
                # Use this line if you want to visualize the weights
                self.Linear[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            # Use this line if you want to visualize the weights
            self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        if self.use_RevIN:
            self.revin = RevIN(self.channels)

    def forward(self, X):
        # x: [Batch, Input length, Channel]
        # Only to train the Linear model
        if self.normalize:
            seq_last = X[:,-1:,:].detach()
            X = X - seq_last

        if self.use_RevIN:
            X = self.revin(X, 'norm')

        output = torch.zeros([X.size(0),self.pred_len,X.size(2)],dtype=X.dtype).to(X.device)
        if self.individual:
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](X[:,:,i])
        else:
            output = self.Linear(X.permute(0,2,1)).permute(0,2,1)
        
        if self.use_RevIN:
            output = self.revin(output, 'denorm')

        if self.normalize:
            output = output + seq_last

        return output # [Batch, Output length, Channel]

    def train(self, X, y):
        # X: [dataset size, Input length, Channel]
        # y: [dataset size, Output length, Channel]
        X, y = X.to(self.device), y.to(self.device)
        if self.normalize:
            seq_last = X[:,-1:,:]
            X = X - seq_last
            y = y - seq_last
        
        # Get predictions of NLinear
        output = torch.zeros([X.size(0),self.pred_len,X.size(2)],dtype=X.dtype).to(X.device)
        with torch.no_grad():
            if self.use_RevIN:
                l_in = self.revin(X, 'norm')
            
            lin_in = X if not self.use_RevIN else l_in
            if self.individual:
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](lin_in[:,:,i].float())
            else:
                output = self.Linear(lin_in.permute(0,2,1).float()).permute(0,2,1)

            if self.use_RevIN:
                output = self.revin(output, 'denorm')
        
        y = y - output
        if self.use_sigmoid:
            y = nn.functional.sigmoid(y)

        if self.treelb > 0:
            if self.lb_data == '0':
                if self.normalize:
                    X += seq_last
                X = torch.cat((X[:,-self.treelb:,:], output), dim=1)
            elif self.lb_data == 'N':
                X = torch.cat((X[:,-self.treelb:,:], output), dim=1)
            else:
                X = torch.cat((lin_in[:,-self.treelb:,:], output), dim=1)
        else:
            X = output
        
        X, y = X.cpu().detach().numpy(), y.cpu().detach().numpy()

        self.Tree = []
        for i in range(self.channels):
            dtrain = lgb.Dataset(X[:,:,i])
            def multi_mse(y_hat, dtrain): 
                """Based on the simplified MSE commonly used in GBM models"""
                y_true = y[:,:, i]
                grad = y_hat - y_true
                hess = np.ones_like(y_true)
                return grad.flatten("F"), hess.flatten("F")
            
            def pseudo_huber(y_hat, dtrain):
                y_true = y[:,:, i]
                d = (y_hat - y_true)
                h = 1  #h is delta 1 = huber loss
                scale = 1 + (d / h) ** 2
                scale_sqrt = np.sqrt(scale)
                grad = d / scale_sqrt 
                hess = 1 / scale / scale_sqrt 

                return grad, hess
            
            def mixed_loss(y_hat, dtrain):
                # MSEloss
                y_true = y[:,:, i]
                grad1 = y_hat - y_true
                hess1 = np.ones_like(y_true)
                
                # Huberloss
                scale = 1 + grad1 ** 2
                scale_sqrt = np.sqrt(scale)
                grad2 = grad1 / scale_sqrt 
                hess2 = 1 / scale / scale_sqrt 

                return 0.5 * (grad1+grad2), 0.5 * (hess1+hess2)

            if self.tree_loss == 'Huber':
                loss_func = pseudo_huber
            elif self.tree_loss == 'Mixed':
                loss_func = mixed_loss
            else:
                loss_func = multi_mse

            self.Tree.append(
                lgb.train(
                    train_set=dtrain,
                    params = {
                        "boosting": "gbdt",
                        "objective": loss_func,
                        "num_class": self.pred_len,
                        "num_threads": self.num_jobs,
                        "num_leaves": self.num_leaves,
                        "learning_rate": self.tree_lr,
                        "num_iterations": self.tree_iter,
                        #"min_data_in_leaf": 20,
                        "force_col_wise":True,
                        "data_sample_strategy": "goss",
                        "path_smooth": self.psmooth,
                        "random_seed": 7,
                        "verbose": 1
                    },
                )
            )


    def predict(self, X):
        X = X.to(self.device)
        if self.normalize:
            seq_last = X[:,-1:,:]
            X = X - seq_last
        
        
        # Get predictions of Linear
        X = X.to(self.device)
        output = torch.zeros([X.size(0),self.pred_len,X.size(2)],dtype=X.dtype).to(X.device)
        with torch.no_grad():
            if self.use_RevIN:
                l_in = self.revin(X, 'norm') 
            
            lin_in = X if not self.use_RevIN else l_in
            if self.individual:
                for i in range(self.channels):
                    output[:,:,i] = self.Linear[i](lin_in[:,:,i].float())
            else:
                output = self.Linear(lin_in.permute(0,2,1).float()).permute(0,2,1)
            
            if self.use_RevIN:
                output = self.revin(output, 'denorm') 
        
        if self.treelb > 0:
            if self.lb_data == '0':
                if self.normalize:
                    X += seq_last
                X = torch.cat((X[:,-self.treelb:,:], output), dim=1)
            elif self.lb_data == 'N':
                X = torch.cat((X[:,-self.treelb:,:], output), dim=1)
            else:
                X = torch.cat((lin_in[:,-self.treelb:,:], output), dim=1)
        else:
            X = output

        X, output = X.cpu().detach().numpy(), output.cpu().detach(), 

        output2 = torch.zeros([output.size(0),self.pred_len,output.size(2)],dtype=output.dtype).to(output.device)
        for i in range(self.channels):
            dtest = X[:,:,i]
            output2[:,:,i] = torch.tensor(self.Tree[i].predict(dtest, num_threads=10), dtype=torch.double)

        if self.use_sigmoid:
            eps=1e-7 # to prevent nan errors when log(0)
            output2 = torch.clamp(output2, min=eps, max=1-eps)
            output2 = torch.log(output2) - torch.log(1-output2)

        if self.normalize:
            seq_last = seq_last.cpu()
            output = output + seq_last

        return output + output2