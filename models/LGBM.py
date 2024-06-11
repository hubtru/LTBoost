import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Multi-Output LightGBM Regression (Expert models)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in 
        
        self.Tree = []
        self.tree_lr = configs.tree_lr
        self.tree_loss = configs.tree_loss
        self.num_leaves = configs.num_leaves
        self.tree_iter = configs.tree_iter
        self.psmooth = configs.psmooth
        self.num_jobs = configs.num_jobs

        self.normalize = configs.normalize

    def train(self, X, y):
        # X: [dataset size, Input length, Channel]
        # y: [dataset size, Output length, Channel]
        if self.normalize:
            seq_last = X[:,-1:,:]
            X = X - seq_last
            y = y - seq_last
        
        y = y.numpy()
        for i in range(self.channels):
            dtrain = lgb.Dataset(X[:,:,i])
            def multi_mse(y_hat, dtrain): 
                """Based on the simplified MSE commonly used in GBM models"""
                y_true = y[:,:, i]
                grad = y_hat.reshape(y_true.shape, order="F") - y_true
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
                        "data_sample_strategy": "goss",
                        "path_smooth": self.psmooth,
                        "force_col_wise":True,
                        "random_seed": 7,
                        "verbose": 1
                    }
                )
            )


    def predict(self, X):
        output = torch.zeros([X.size(0),self.pred_len,X.size(2)],dtype=X.dtype).to(X.device)
        if self.normalize:
            seq_last = X[:,-1:,:]
            X = X - seq_last
        
        X = X.numpy()
        for i in range(self.channels):
            dtest = X[:,:,i]
            output[:,:,i] = torch.tensor(self.Tree[i].predict(dtest), dtype=torch.double)

        if self.normalize:
            output = output + seq_last

        return output
