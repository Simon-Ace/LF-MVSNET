"""
all parameters are in here
"""

import os

# dataset
LF_train_dir = '/home/dell/User/shuo/dataset/HCI4D/additional/'  #????
LF_test_dir = '/home/dell/User/shuo/dataset/HCI4D/training/'

# create a param object
param = type('',(object,),{})

param.model_name = 'epinet'
param.is_continue = False   # True means you can load the lastest checkpoint to continue training

param.batch_size = 16
param.input_size = 25             #23+2   # 经过一堆卷积之后23x23 -> 1x1，改成：25x25 -> 3x3
param.model_conv_depth = 7
# param.output_size = param.input_size - 22
param.output_size = param.input_size - (param.model_conv_depth + 1) * 2 - 6    # 有几个valid的conv，就得减几
param.model_filter_nums = 70
param.model_learning_rate = 1e-4  #1e-4
param.steps_per_epoch = 5000
param.worker_num = 2 # ???

# NEW
param.feature_filters_num = 32  # 特征提取部分卷积核数量
param.featrue_depth = 3         # N个ResBlock，每个ResBlock里面两个conv2D
param.cost_filter_depth = 7     # cost volume filter部分网络层数
param.input_pic_num = 5             # 输入图片数量


param.idx_90d = [76, 67, 58, 49, 40, 31, 22, 13, 4]  # 垂直方向
param.idx_0d = [36, 37, 38, 39, 40, 41, 42, 43, 44]  # 水平方向的index
param.idx_45d = [72, 64, 56, 48, 40, 32, 24, 16, 8]  # 左下->右上
param.idx_m45d = [0, 10, 20, 30, 40, 50, 60, 70, 80]  # 左上->右下

param.idx_5pic = [40, 4, 36, 44, 76]     #把中心视角放最前面

param.input_img_size = 512
param.output_img_size = param.input_img_size-22

param.trainset_dirs = [
    os.path.join(LF_train_dir,'antinous/'),os.path.join(LF_train_dir,'boardgames/'),
    os.path.join(LF_train_dir,'dishes/'),os.path.join(LF_train_dir,'greek/'),
    os.path.join(LF_train_dir,'medieval2/'),os.path.join(LF_train_dir,'pens/'),
    os.path.join(LF_train_dir,'pillows/'),os.path.join(LF_train_dir,'platonic/'),
    os.path.join(LF_train_dir,'rosemary/'),os.path.join(LF_train_dir,'table/'),
    os.path.join(LF_train_dir,'tomb/'),os.path.join(LF_train_dir,'tower/'),
    os.path.join(LF_train_dir,'town/')
]

# for output
param.logfile_dir = 'out/' + param.model_name + '_train.txt'
param.iter_dir = 'out/'+param.model_name +'/iteration/'
param.checkpoint_dir = 'out/'+param.model_name +'/checkpoints/'
param.prediction_dir = 'out/'+param.model_name +'/prediction/'




