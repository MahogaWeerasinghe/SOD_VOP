from __future__ import division
from scipy.__config__ import show
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch import optim
import torchvision.transforms as transforms


import numpy as np
import os, sys, argparse
from datetime import datetime
from scipy import misc
from PIL import Image
sys.path.append('..')

from data import get_loader, test_dataset
from metric import AvgMeter, cal_mae, cal_maxF, cal_sm, cal_acc, cal_meanF, cal_ber
from model.ResNet_models import DCN

parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--channel', type=int, default=128, help='channel number of convolutional layers in decoder')

config = parser.parse_args()
print(config)

np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)

data_path = '/content/drive/MyDrive/Mahoga/Datasets/'

# evolution metrices created from metric.py
accuracy = cal_acc() # accuracy <- doesn't sure that code
MAE = cal_mae() # Mean absoulte error
FM = cal_meanF() # F-measure
SM = cal_sm() # Structure Similarity




model = DCN(channel=config.channel)
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/SOD-VOP_withSaliencyMap/DCN/Trained_Models/DCN_2ndTime.pth', map_location=torch.device('cpu') ))

valset = ['ECSSD']
model.eval()
for dataset in valset:
    save_path = '/content/drive/MyDrive/Mahoga/Results/SOD_VOP/test-metric-ECSSD/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = data_path + dataset + '/ECSSD-images/'
    gt_root = data_path + dataset + '/ECSSD-mask/'
    test_loader = test_dataset(image_root, gt_root, config.trainsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        #print("image : ", image)
        #print("gt : ", gt)
        file = save_path + name + '.png'
        gt = np.array(gt).astype('float')
        gt = gt / (gt.max() + 1e-8)
        if torch.cuda.is_available():
            image = Variable(image).cuda()
        else:
            image = Variable(image)

        res = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        # metrices
        accuracy.update(res,gt)
        MAE.update(res,gt)
        FM.update(res,gt)
        SM.update(res,gt)

        res = Image.fromarray(np.uint8(255*res)).convert('RGB')
        res.save(save_path+name+'.png')
        
      

# print metrices
print('accuracy : ', accuracy.show())
print('mean MAE : ', MAE.show())
print('F-Measure : ', FM.show())
print('Structure Similarity : ', SM.show())
