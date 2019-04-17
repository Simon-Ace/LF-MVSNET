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
param.input_size = 64 #23+2 # ？？？？
param.output_size = param.input_size - 22
param.model_conv_depth = 7
param.model_filter_nums = 70
param.model_learning_rate = 1e-4
param.steps_per_epoch = 4000
param.worker_num = 2 # ???

# param.idx_90d = [76, 67, 58, 49, 40, 31, 22, 13, 4]  # 垂直方向
# param.idx_0d = [36, 37, 38, 39, 40, 41, 42, 43, 44]  # 水平方向的index
# param.idx_45d = [72, 64, 56, 48, 40, 32, 24, 16, 8]  # 左下->右上
# param.idx_m45d = [0, 10, 20, 30, 40, 50, 60, 70, 80]  # 左上->右下

# param.idx_90d = [76, 58, 40, 22, 4]  # 垂直方向
# param.idx_0d = [36, 38, 40, 42, 44]  # 水平方向的index
# param.idx_45d = [72, 56, 40, 24, 8]  # 左下->右上
# param.idx_m45d = [0, 20, 40, 60, 80]  # 左上->右下

# 效果不好
# param.idx_90d = [76, 40, 4]  # 垂直方向
# param.idx_0d = [36, 40, 44]  # 水平方向的index
# param.idx_45d = [72, 40, 8]  # 左下->右上
# param.idx_m45d = [0, 40, 80]  # 左上->右下

# 效果不好
# param.idx_90d = [76, 58, 40]  # 垂直方向
# param.idx_0d = [36, 38, 40]  # 水平方向的index
# param.idx_45d = [72, 56, 40]  # 左下->右上
# param.idx_m45d = [0, 20, 40]  # 左上->右下

# 十字坐标
param.idx_90d = [4, 40]
param.idx_0d = [76, 40]
param.idx_45d = [36, 40]
param.idx_m45d = [44, 40]

param.each_row_pic_num = len(param.idx_90d)
param.disp = 15

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
param.logfile_dir = 'out_cost_argmin_1/' + param.model_name + '_train.txt'
param.iter_dir = 'out_cost_argmin_1/'+param.model_name +'/iteration/'
param.checkpoint_dir = 'out_cost_argmin_1/'+param.model_name +'/checkpoints/'
param.prediction_dir = 'out_cost_argmin_1/'+param.model_name +'/prediction/'
