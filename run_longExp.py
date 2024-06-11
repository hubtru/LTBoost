import argparse
import os
import time
import torch
from exp.exp_main import Exp_Main
from exp.exp_lgbm import Exp_LGBM
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='LTBoost for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Linear',
                    help='model name, options: [Linear, DLinear, NLinear, LightGBM, LTBoost]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='Number of variates')

# Linear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='linear loss function, options:[MAE, MSE, Custom]')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# LightGBM
parser.add_argument('--tree_lr', type=float, default=0.01, help='LGBM learning rate')
parser.add_argument('--tree_loss', type=str, default='MSE', help='Tree loss function, options:[Huber, MSE, Mixed]')
parser.add_argument('--num_leaves', type=int, default=2, help='Number of leaves')
parser.add_argument('--tree_iter', type=int, default=200, help='Number of iterations')
parser.add_argument('--psmooth', type=int, default=0, help='LGBM Pathsmoothing parameter')
parser.add_argument('--num_jobs', type=int, default=10, help='Number of parallel jobs for LGBM/LTBoost')

# LTBoost
parser.add_argument('--normalize', action='store_true', default=False, help='Use N-Normalization')
parser.add_argument('--use_revin', action='store_true', default=False, help='Use RevIN-Normalization')
parser.add_argument('--use_sigmoid', action='store_true', default=False, help='Use Sigmoid residual normalization') # Only used for weather

parser.add_argument('--tree_lb', type=int, default=336, help='LGBM lookback window when using LTBoost') # Similar to seq_len for Linear/LGBM
parser.add_argument('--lb_data', type=str, default='RIN', help='Which data should lgbm use, options:[0, N, RIN]')
# 0: raw data, N: N-Normalized, RIN: N+RevIN Normalized

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

if args.model in ["LightGBM", "LTBoost"]:
    Exp = Exp_LGBM
else:
    Exp = Exp_Main


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.des, ii)

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        start_time = time.time()
        exp.predict(setting, True)
        predict_time = time.time()

        print("Predicting Time: ", predict_time - start_time)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        start_time = time.time()
        exp.test(setting, test=1)
        test_time = time.time()

        print("Testing Time: ", test_time - start_time)

    torch.cuda.empty_cache()
