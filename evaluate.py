from data_process.func_pfm import *
from utils import get_diff_badpixel_1pic
from param import *
import os


if __name__ == '__main__':
    for dir in os.listdir(LF_test_dir):
        print("load ",str(dir))
        label = np.float32(read_pfm(os.path.join(LF_test_dir,dir)+'/gt_disp_lowres.pfm'))
        result = np.float32(read_pfm(param.prediction_dir+dir+'291.pfm'))
        if param.output_img_size != param.input_img_size:
            idx = int((512-param.output_img_size)/2)
            label = np.copy(label[idx:-idx,idx:-idx])
        mse, bp = get_diff_badpixel_1pic(result,label)
        print("mse: %f, bad_pixel:%f"%(mse,bp))
        print("="*30)