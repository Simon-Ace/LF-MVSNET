from param import *
from data_process.utils import read_data, stack_data
import os
# from my_model import get_model
from mycnn.epimodel import get_model
from data_process.func_pfm import write_pfm
import re

'''GPU setting'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

subdirs = os.listdir(LF_test_dir)


# load model
# model = get_model(input_shape=(512, 512, 9), filters_count=70, conv_depth=7, learning_rate=1e-4)
# model = get_model(input_shape=(512, 512, param.each_row_pic_num), filters_count=70, conv_depth=7, learning_rate=1e-4)
model = get_model(input_shape=(512, 512, param.disp * 2 + 1), filters_count=70, conv_depth=7, learning_rate=1e-4)

# load the latest weights
ckps = os.listdir(param.checkpoint_dir)
ckps.sort()
latest_ckp = ckps[-1]
iter = int(re.findall(r'\d+', latest_ckp)[0])

model.load_weights(os.path.join(param.checkpoint_dir,latest_ckp))

if not os.path.exists(param.prediction_dir):
    os.makedirs(param.prediction_dir)

for dir in subdirs:
    raw_90d, raw_0d, raw_45d, raw_m45d, _ = read_data([os.path.join(LF_test_dir,dir)],param.idx_90d,param.idx_0d,param.idx_45d,param.idx_m45d,param.input_img_size)
    input_90d,input_0d,input_45d,input_m45d = stack_data(raw_90d, raw_0d,raw_45d,raw_m45d)

    result = model.predict(x=[input_90d,input_0d,input_45d,input_m45d],batch_size=1)

    write_pfm(result[0,:,:,0],param.prediction_dir+dir+str(iter)+'.pfm')
    print("Prediction %s, Done." % dir)
