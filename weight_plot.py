import torch
import numpy as np
import os 
import pickle
import matplotlib.pyplot as plt

model_name = 'LTBoost'
for root, dirs, files in os.walk("checkpoints"):
    for name in files:
        model_path = os.path.join(root, name)
        if model_name not in model_path:
            continue
        if model_name == 'LTBoost':
            weights = pickle.load(open(model_path, 'rb')).state_dict()
        else:
            weights = torch.load(model_path,map_location=torch.device('cpu'))
        
        weights_list = {}
        if model_name in ['Linear', 'NLinear', 'LTBoost']:
            weights_list['Full'] = weights['Linear.weight'].cpu().numpy()
        else:
            weights_list['seasonal'] = weights['Linear_Seasonal.weight'].numpy()
            weights_list['trend'] = weights['Linear_Trend.weight'].numpy()

        save_root = 'weights_plot/%s'%root.split('\\')[1]
        #save_root = 'weights_plot/%s'%root.split('/')[1]
        if not os.path.exists('weights_plot'):
            os.mkdir('weights_plot')
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        for w_name,weight in weights_list.items():
            fig,ax=plt.subplots()
            ax.set_ylabel('Forecastingstep')
            ax.set_xlabel('Lookbackstep')
            im=ax.imshow(weight,cmap='plasma_r')
            fig.colorbar(im,pad=0.03)
            plt.savefig(os.path.join(save_root,w_name + '.pdf'),dpi=500)
            plt.close()
 