from param import *
import numpy as np
#from .func_pfm import *
import imageio

from data_process.func_pfm import *

# 读取图片，MY，5张图
def read_any_num_data(dirs, img_size, img_num):
    '''
    :param dirs: 文件目录
    :param img_size: 图片大小（长or宽）
    :param img_num: 图片数量
    :return raw_data_list（n张输入图片）one item:(512, 512, 3, 13)
    :return raw_label（深度图）(512, 512, 13)
    '''
    raw_data_list = []
    for i in range(len(img_num)):
        raw_data_list.append(np.zeros(shape=(img_size, img_size, 3, len(dirs)), dtype=np.float32))
    raw_label = np.zeros(shape=(img_size, img_size, len(dirs)), dtype=np.float32)

    i_scence = 0
    for dir in dirs:
        print("loading...", dir)
        for i in range(len(img_num)):
            raw_data_list[i][:, :, :, i_scence] = np.float32(
                imageio.imread(dir + '/input_Cam0%02d.png' % (param.idx_5pic[i])))

        raw_label[:, :, i_scence] = np.array(read_pfm(dir + '/gt_disp_lowres.pfm'), dtype=np.float32)
        i_scence += 1
    return raw_data_list, raw_label

# 图片灰度化 512x512的图
def stack_any_num_data(raw_data_list):
    """
    stack 9 images into 9channels using fixed gray scale, use for testing
    for example:
        input: raw_data_0d(512, 512, 3,9,n_scences)
        output: stack_data_0d(n_scences, 512, 512, 9 )
    """
    R = 0.299
    G = 0.587
    B = 0.114

    stack_data_list = []
    for i in range(len(raw_data_list)):
        # H x W x 1(channel) x 文件夹数目
        stack_data_list.append(np.zeros((raw_data_list[i].shape[0], raw_data_list[i].shape[1], 1,
                                         raw_data_list[i].shape[-1]), dtype=np.float32))

        stack_data_list[i][:,:,0,:] = (raw_data_list[i][:, :, 0, :] * R + raw_data_list[i][:, :, 1, :] * G
                                      + raw_data_list[i][:, :, 2, :] * B) # / 255

        # (512,512,1,n_scences) -> (n_scences,512,512,1)
        stack_data_list[i] = np.transpose(stack_data_list[i], (3, 0, 1, 2))

    return stack_data_list
    '''
        stack_data_0d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], 9, raw_data_0d.shape[-1]), dtype=np.float32)
    stack_data_90d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], 9, raw_data_0d.shape[-1]), dtype=np.float32)
    stack_data_45d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], 9, raw_data_0d.shape[-1]), dtype=np.float32)
    stack_data_m45d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], 9, raw_data_0d.shape[-1]), dtype=np.float32)

    stack_data_any_num = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], 9, raw_data_0d.shape[-1]), dtype=np.float32)


    for i in range(9):
        stack_data_0d[:, :, i, :] = raw_data_0d[:, :, 0, i, :] * R + raw_data_0d[:, :, 1, i, :] * G + raw_data_0d[:, :,
                                                                                                      2, i, :] * B
        stack_data_90d[:, :, i, :] = raw_data_90d[:, :, 0, i, :] * R + raw_data_90d[:, :, 1, i, :] * G + raw_data_90d[:,
                                                                                                         :, 2, i, :] * B
        stack_data_45d[:, :, i, :] = raw_data_45d[:, :, 0, i, :] * R + raw_data_45d[:, :, 1, i, :] * G + raw_data_45d[:,
                                                                                                         :, 2, i, :] * B
        stack_data_m45d[:, :, i, :] = raw_data_m45d[:, :, 0, i, :] * R + raw_data_m45d[:, :, 1, i,
                                                                         :] * G + raw_data_m45d[:, :, 2, i, :] * B

    stack_data_0d, stack_data_90d, stack_data_45d, stack_data_m45d = stack_data_0d / 255, stack_data_90d / 255, stack_data_45d / 255, stack_data_m45d / 255

    # (512,512,9,n_scences) -> (n_scences,512,512,9)
    stack_data_0d = np.transpose(stack_data_0d, (3, 0, 1, 2))
    stack_data_90d = np.transpose(stack_data_90d, (3, 0, 1, 2))
    stack_data_45d = np.transpose(stack_data_45d, (3, 0, 1, 2))
    stack_data_m45d = np.transpose(stack_data_m45d, (3, 0, 1, 2))
    return stack_data_90d, stack_data_0d, stack_data_45d, stack_data_m45d
    '''

def random_gray_resized_crop_any_num(w_start, h_start, scale, scence_id, data, R, G, B):
    """
    :param w_start: int, index of width to start crop
    :param h_start: int, index of height to start crop
    :param scale: int
    :param scence_id:int
    :param data: ndarray(512,512,3,n_scence)
    :param R: float
    :param G: float
    :param B: float
    :return: croped: ndarray(input_size,input_size,1)
    """
    croped = data[w_start:w_start + scale * param.input_size:scale, h_start:h_start + scale * param.input_size:scale,
             0, scence_id] * R \
             + data[w_start:w_start + scale * param.input_size:scale, h_start:h_start + scale * param.input_size:scale,
               1, scence_id] * G \
             + data[w_start:w_start + scale * param.input_size:scale, h_start:h_start + scale * param.input_size:scale,
               2, scence_id] * B

    # 加个维度（Keras输入的时候必须有三个维度）
    croped = croped.reshape(croped.shape[0], croped.shape[1], 1)
    return croped


def random_resized_crop(w_start, h_start, scale, scence_id, data):
    """

    :param w_start: int, index of width to start crop
    :param h_start: int, index of height to start crop
    :param scale: int
    :param scence_id: int
    :param data: ndarray(512,512,n_scence)
    :return: croped: ndarray(output_size,output_size)
    """
    # croped = data[w_start:w_start+scale*param.input_size:scale,h_start:h_start+scale*param.input_size:scale,scence_id]
    pad = int((param.input_size - param.output_size) / 2)
    # return croped[pad:-pad,pad:-pad] / scale
    croped = data[w_start + scale * pad:w_start + scale * pad + scale * param.output_size:scale,
             h_start + scale * pad:h_start + scale * pad + scale * param.output_size:scale, scence_id]
    return croped / scale


def generate_any_num_train_data(raw_data_list, raw_label):
    """
    :param raw_data_list: ndarray_list each item -> (512,512,3,n_scences)
    :param raw_label: ndarray(512,512,n_scences)
    :return: train_batch_list ???
            train_batch_label ???
    """
    train_batch_list = []
    for raw_i in range(len(raw_data_list)):
        train_batch_list.append(np.zeros((param.input_size, param.input_size, 1, param.batch_size), dtype=np.float32))
    train_batch_label = np.zeros((param.output_size, param.output_size, param.batch_size), dtype=np.float32)

    all_scences = np.array(list(range(len(param.trainset_dirs))))

    for batch_i in range(param.batch_size):
        while True:
            # Variable for gray conversion
            rand_3color = 0.05 + np.random.rand(3)
            rand_3color = rand_3color / np.sum(rand_3color)
            R = rand_3color[0]
            G = rand_3color[1]
            B = rand_3color[2]

            # randomly choose one scences
            scence_id = np.random.choice(all_scences)

            # gray conversion and stack, random_size_crop
            kk = np.random.randint(17)
            if (kk < 8):
                scale = 1
            elif (kk < 14):
                scale = 2
            elif (kk < 17):
                scale = 3
            x_start = np.random.randint(0, 512 - scale * param.input_size)
            y_start = np.random.randint(0, 512 - scale * param.input_size)

            test = random_gray_resized_crop_any_num(x_start, y_start, scale, scence_id, raw_data_list[0], R, G, B)

            # raw_data_list[0]是中心视角的图
            center_img = random_gray_resized_crop_any_num(x_start, y_start, scale, scence_id, raw_data_list[0], R, G, B)
            center_img = center_img[:, :, 0] / 255
            sum_diff = np.sum(
                np.abs(center_img - np.squeeze(center_img[int(0.5 * param.input_size), int(0.5 * param.input_size)])))

            # 去除纹理不明显的区域
            if sum_diff < 0.01 * param.input_size * param.input_size:
                continue
            else:
                #TODO: random_gray_resized_crop_any_num() 所有图像使用这个函数，注意输出的时候数据结构
                # crop_list = []
                for raw_idx in range(len(raw_data_list)):
                    crop_temp = random_gray_resized_crop_any_num(
                                                        x_start, y_start, scale, scence_id, raw_data_list[raw_idx], R, G, B)
                    train_batch_list[raw_idx][:, :, :, batch_i] =  crop_temp

                crop_label = random_resized_crop(x_start, y_start, scale, scence_id, raw_label)
                train_batch_label[:, :, batch_i] = crop_label

                break

    for raw_idx in range(len(raw_data_list)):
        train_batch_list[raw_idx] = np.transpose(train_batch_list[raw_idx], (3, 0, 1, 2)) / 255

    train_batch_label = np.transpose(train_batch_label, (2, 0, 1))


    return train_batch_list, train_batch_label





if __name__ == '__main__':
    raw_data_list, raw_label = read_any_num_data(param.trainset_dirs, param.input_img_size, param.idx_5pic)

    #write_pfm(raw_label[:,:,0], './123.pfm')
    '''
    vis2 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    vis2[:, :, :] = raw_data_list[0][:, :, :, 0]
    imageio.imwrite(r'1.jpg', vis2)
    '''

    train512_data_list = stack_any_num_data(raw_data_list)

    '''
    vis2 = np.zeros(shape=(512, 512, 1), dtype=np.uint8)
    vis2[:, :, :] = train512_data_list[0][0, :, :, :]
    imageio.imwrite(r'../mid_out/1.jpg', vis2)
    '''
    train_batch_list, train_batch_label = generate_any_num_train_data(raw_data_list, raw_label)

    print(1)
