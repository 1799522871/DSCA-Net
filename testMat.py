import scipy.io as scio
import os
import os.path as osp

base_dir = '/root/data/DSCA-Net/data/'

data_path_mat = osp.join(base_dir,'PaviaU.mat')
gt_path_mat = osp.join(base_dir,'PaviaU_gt.mat')
data = scio.loadmat(data_path_mat)
gt = scio.loadmat(gt_path_mat)

print(data)  # 610*340*103   103个光谱波段
print(gt)    # 610*340
print('========================')