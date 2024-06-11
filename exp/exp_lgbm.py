from data_provider.data_factory import data_provider
from models import LGBM, LTBoost
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle

import warnings
import numpy as np

warnings.filterwarnings('ignore')

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 0.5 * F.mse_loss(x, y) + 0.5 * F.l1_loss(x, y)

class Exp_LGBM:
    def __init__(self, args):
        self.args = args
        model_dict = {
            'LightGBM': LGBM,
            'LTBoost': LTBoost
        }
        self.model = model_dict[args.model].Model(self.args)
        self.device = 'cuda' if args.use_gpu else 'cpu'
        self.dataset_name = self.args.data_path[:-4]

    def _get_data(self, flag):
        dataset, data_loader = data_provider(self.args, flag)
        return dataset, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif self.args.loss == 'Custom':
            criterion = CustomLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def validate(self, vali_loader, criterion):
        # Equivalent of vali method in exp_main 
        # Used for Linear
        # Returns average loss
        total_loss = []

        with torch.no_grad():
            for batch_x, batch_y, _, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:,:].float().to(self.device)
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:,:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        return total_loss

    def vali(self, vali_loader, loss_function):
        # Run predictions on the given data and calculate the loss
        # Used for LTBoost
        # Returns MAE and MSE

        X_batches, y_batches = [], [],
        for batch_X, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            X_batches.append(batch_X.float())
            y_batches.append(batch_y[:, -self.args.pred_len:,:].float())
        X, y = torch.cat(X_batches, dim=0), torch.cat(y_batches, dim=0)

        outputs = self.model.predict(X)
        outputs = outputs.detach().cpu().numpy()
        y = y.numpy()
        mae, mse, _, _, _, _, _ = loss_function(outputs, y)

        return mae,mse

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # If hybrid then train Linear
        if self.args.model == "LTBoost":
            time_now = time.time()
            train_steps = len(train_loader)
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()
            self.model = self.model.to(self.device)
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                epoch_time = time.time()
                for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:,:]
                    batch_y = batch_y[:, -self.args.pred_len:,:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                if not self.args.train_only:
                    vali_loss = self.validate(vali_loader, criterion)
                    test_loss = self.validate(test_loader, criterion)

                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                else:
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                        epoch + 1, train_steps, train_loss))
                    early_stopping(train_loss, self.model, path)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Get full dataset for LGBM
        X_batches, y_batches = [], [],
        for batch_X, batch_y, _, _ in train_loader:
            X_batches.append(batch_X.float())
            y_batches.append(batch_y[:, -self.args.pred_len:,:].float())
        X, y = torch.cat(X_batches, dim=0), torch.cat(y_batches, dim=0)

        # Train LGBM
        time_start = time.time()
        print("Start LGBM Training")
        self.model.train(X, y)
        time_train = time.time()

        print("Predicting")
        outputs = self.model.predict(X)
        time_predict = time.time()
        outputs = outputs.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        train_mae, train_mse, _, _, _, _, _ = metric(outputs, y)
        
        print("Training time: {0:.2f}".format(time_train - time_start))
        print("(Train) Prediction time: {0:.2f}".format(time_predict - time_train))

        if not self.args.train_only:
            # Run validation on validation and test dataset
            vali_mae, vali_mse = self.vali(vali_loader, metric)
            test_mae, test_mse = self.vali(test_loader, metric)

            print("(mse/mae) Train: {0:.7f}/{1:.7f} Vali: {2:.7f}/{3:.7f} Test: {4:.7f}/{5:.7f}".format(
                train_mse, train_mae, vali_mse, vali_mae, test_mse, test_mae))
        else:
            print("Train mae: {0:.7f} Train mse: {1:.7f}".format(train_mse, train_mae))


        best_model_path = path + '/' + 'checkpoint.pth'
        pickle.dump(self.model, open(best_model_path, 'wb'))

        return self.model


    def test(self, setting, test=False):
        # Test the trained model on the test dataset
        _, test_loader = self._get_data(flag='test')

        # Load the model
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model = pickle.load(open(best_model_path, 'rb'))

        folder_path = os.path.join(self.args.root_path, "..", "test_results", setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Run predictions
        X_batches, y_batches = [], [],
        for batch_X, batch_y, _, _ in test_loader:
            X_batches.append(batch_X.float())
            y_batches.append(batch_y[:, -self.args.pred_len:,:].float())
        X, y = torch.cat(X_batches, dim=0), torch.cat(y_batches, dim=0)
        
        test_start = time.time()
        outputs = self.model.predict(X)
        test_end = time.time()
        print("(Test) Prediction time: {0:.2f}. Number of samples: {1}".format(test_end - test_start, len(X)))
        ms = (test_end - test_start) * 1000 / len(X)
        print("{}ms/sample".format(ms))
        
        outputs = outputs.detach().cpu().numpy()
        y = y.numpy()
        # result save
        folder_path = os.path.join(self.args.root_path, "..", "results", setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Compute metrics and save them
        mae, mse, rmse, mape, mspe, rse, corr = metric(outputs, y)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # Save raw predictions
        #np.save(os.path.join(folder_path, 'pred.npy'), outputs)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model = pickle.load(open(best_model_path, 'rb'))

        # Run predictions
        X_batches = []
        for batch_X, _, _, _ in pred_loader:
            X_batches.append(batch_X.float())
        X = torch.cat(X_batches, dim=0)
        preds = self.model.predict(X)
        preds = preds.numpy()
        
        preds = np.concatenate(preds, axis=0)
        if pred_data.scale:
            preds = pred_data.inverse_transform(preds)

        # Save the predictions results
        #folder_path = os.path.join(self.args.root_path, "..", "results", setting)
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        #np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        #pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False) 
        return 