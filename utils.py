import numpy as np
import imageio
from param import *
from data_process.utils_my import read_any_num_data

def get_diff_badpixel(train_output,train_label,iter,dir_save):
    sz = len(train_output)
    train_label = np.transpose(train_label,(2,0,1))
    train_output = np.squeeze(train_output)
    train_label482 = train_label[:,15:-15, 15:-15]
    #推出来的公式：512 - 2(model_conv_depth+1) - 2pad = 482
    pad_width = 14 - param.model_conv_depth
    train_output482 = train_output[:, pad_width:-pad_width, pad_width:-pad_width]

    train_diff = np.abs(train_output482 - train_label482)
    train_bp = (train_diff >= 0.07)

    result = np.zeros((2*482,sz*482),np.uint8)
    result[0:482,:] = np.uint8(25*np.reshape(np.transpose(train_label482,(1,0,2)),(482,sz*482))+100)
    result[482:2 * 482, :] = np.uint8(25 * np.reshape(np.transpose(train_output482, (1, 0, 2)), (482, sz * 482)) + 100)

    imageio.imsave(dir_save+ '/train_iter%05d.jpg' % (iter), np.squeeze(result))

    return train_diff, train_bp

def get_diff_badpixel_padding(train_output,train_label,iter,dir_save):
    sz = len(train_output)
    train_label = np.transpose(train_label,(2,0,1))
    train_output = np.squeeze(train_output)
    train_label482 = train_label[:,15:-15, 15:-15]
    train_output482 = train_output[:,15:-15, 15:-15]

    train_diff = np.abs(train_output482 - train_label482)
    train_bp = (train_diff >= 0.07)

    result = np.zeros((2*482,sz*482),np.uint8)
    result[0:482,:] = np.uint8(25*np.reshape(np.transpose(train_label482,(1,0,2)),(482,sz*482))+100)
    result[482:2 * 482, :] = np.uint8(25 * np.reshape(np.transpose(train_output482, (1, 0, 2)), (482, sz * 482)) + 100)

    imageio.imsave(dir_save+ '/train_iter%05d.jpg' % (iter), np.squeeze(result))

    return train_diff, train_bp

def get_diff_badpixel_1pic(result, label):
    train_diff = np.abs(result-label)
    train_bp = (train_diff >= 0.07)
    mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    bad_pixel_ratio = 100 * np.average(train_bp)

    return mean_squared_error_x100, bad_pixel_ratio

if __name__ == '__main__':
    raw_data_list, raw_label = read_any_num_data(param.trainset_dirs, param.input_img_size, param.idx_5pic)
    train512_output = np.fromfile('test2.dat', dtype=np.float32)
    train512_output = train512_output.reshape(13, 494, 494, 1)

    train_diff, train_bp = get_diff_badpixel(train512_output, raw_label, iter=1, dir_save=param.iter_dir)
    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    training_bad_pixel_ratio = 100 * np.average(train_bp)