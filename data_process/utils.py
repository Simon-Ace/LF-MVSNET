from param import *
import numpy as np
from data_process.func_pfm import *
import imageio


def read_data(dirs, idx_90d, idx_0d, idx_45d, idx_m45d, img_size):
    """
    read raw lf images
    return:
        raw_data_0d(n_scences,512,512,3,9) ->  num_scences, height, width, channel,9 images,
        raw_data_90d
        raw_data_45d
        raw_data_,45d
        raw_label(n_scences,512,512)
    """
    raw_data_90d = np.zeros(shape=(img_size, img_size, 3, param.each_row_pic_num, len(dirs)), dtype=np.float32)
    raw_data_0d = np.zeros(shape=(img_size, img_size, 3, param.each_row_pic_num, len(dirs)), dtype=np.float32)
    raw_data_45d = np.zeros(shape=(img_size, img_size, 3, param.each_row_pic_num, len(dirs)), dtype=np.float32)
    raw_data_m45d = np.zeros(shape=(img_size, img_size, 3, param.each_row_pic_num, len(dirs)), dtype=np.float32)
    raw_label = np.zeros(shape=(img_size, img_size, len(dirs)), dtype=np.float32)

    i_scence = 0
    for dir in dirs:
        print("loading...", dir)
        for idx in range(len(idx_0d)):
            raw_data_90d[:, :, :, idx, i_scence] = np.float32(
                imageio.imread(dir + '/input_Cam0%02d.png' % (idx_90d[idx])))
            raw_data_0d[:, :, :, idx, i_scence] = np.float32(
                imageio.imread(dir + '/input_Cam0%02d.png' % (idx_0d[idx])))
            raw_data_45d[:, :, :, idx, i_scence] = np.float32(
                imageio.imread(dir + '/input_Cam0%02d.png' % (idx_45d[idx])))
            raw_data_m45d[:, :, :, idx, i_scence] = np.float32(
                imageio.imread(dir + '/input_Cam0%02d.png' % (idx_m45d[idx])))
        raw_label[:, :, i_scence] = np.array(read_pfm(dir + '/gt_disp_lowres.pfm'), dtype=np.float32)
        i_scence += 1
    return raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label


def stack_data(raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d):
    """
    stack 9 images into 9channels using fixed gray scale, use for testing
    for example:
        input: raw_data_0d(512, 512, 3,9,n_scences)
        output: stack_data_0d(n_scences, 512, 512, 9 )
    """
    R = 0.299
    G = 0.587
    B = 0.114
    stack_data_0d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], param.each_row_pic_num, raw_data_0d.shape[-1]), dtype=np.float32)
    stack_data_90d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], param.each_row_pic_num, raw_data_0d.shape[-1]), dtype=np.float32)
    stack_data_45d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], param.each_row_pic_num, raw_data_0d.shape[-1]), dtype=np.float32)
    stack_data_m45d = np.zeros((raw_data_0d.shape[0], raw_data_0d.shape[1], param.each_row_pic_num, raw_data_0d.shape[-1]), dtype=np.float32)


    for i in range(param.each_row_pic_num):
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

    # cost volume
    stack_data_disp_0d, stack_data_disp_90d, stack_data_disp_45d, stack_data_disp_m45d = \
        build_cost_volume(stack_data_0d, stack_data_90d, stack_data_45d, stack_data_m45d)

    # return stack_data_90d, stack_data_0d, stack_data_45d, stack_data_m45d
    return stack_data_disp_90d, stack_data_disp_0d, stack_data_disp_45d, stack_data_disp_m45d


def random_gray_resized_crop(w_start, h_start, scale, scence_id, data, R, G, B):
    """
    :param w_start: int, index of width to start crop
    :param h_start: int, index of height to start crop
    :param scale: int
    :param scence_id:int
    :param data: ndarray(512,512,3,9,n_scence)
    :param R: float
    :param G: float
    :param B: float
    :return: croped: ndarray(input_size,input_size,9)
    """
    croped = data[w_start:w_start + scale * param.input_size:scale, h_start:h_start + scale * param.input_size:scale, 0,
             :, scence_id] * R \
             + data[w_start:w_start + scale * param.input_size:scale, h_start:h_start + scale * param.input_size:scale,
               1, :, scence_id] * G \
             + data[w_start:w_start + scale * param.input_size:scale, h_start:h_start + scale * param.input_size:scale,
               2, :, scence_id] * B
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

# TODO: 构建的时候要加shape
def build_cost_volume(train_batch_0d, train_batch_90d, train_batch_45d, train_batch_m45d):
    # TODO: 构架cost volume
    '''
    0d  -> 4 40      90d -> 76 40
    45d -> 36 40    m45d -> 44 40
    '''
    train_batch_disp_0d = np.zeros((train_batch_0d.shape[0], train_batch_0d.shape[1], train_batch_0d.shape[2], param.disp * 2 + 1),
                                   dtype=np.float32)
    train_batch_disp_90d = np.zeros((train_batch_90d.shape[0], train_batch_90d.shape[1], train_batch_90d.shape[2], param.disp * 2 + 1),
                                    dtype=np.float32)
    train_batch_disp_45d = np.zeros((train_batch_45d.shape[0], train_batch_45d.shape[1], train_batch_45d.shape[2], param.disp * 2 + 1),
                                    dtype=np.float32)
    train_batch_disp_m45d = np.zeros((train_batch_m45d.shape[0], train_batch_m45d.shape[1], train_batch_m45d.shape[2], param.disp * 2 + 1),
                                     dtype=np.float32)

    # #测试图像分割
    # vis1 = np.zeros(shape=(param.input_size, param.input_size, 1), dtype=np.float32)
    # vis2 = np.zeros(shape=(param.input_size, param.input_size, 1), dtype=np.float32)
    # vis1[:-15, :, 0] = train_batch_0d[0, :-15, :, 0]
    # vis2[:, :, 0] = train_batch_0d[0, :, :, 0]
    # imageio.imwrite(r'1.jpg', vis1)
    # imageio.imwrite(r'2.jpg', vis2)

    # 先Hight后width
    for k in range(-param.disp, param.disp + 1):
        if k < 0:
            train_batch_disp_0d[:, :k, :, k + param.disp] = train_batch_0d[:, :k, :, 0] - train_batch_0d[:, -k:, :, 1]
            train_batch_disp_90d[:, :k, :, k + param.disp] = train_batch_90d[:, :k, :, 0] - train_batch_90d[:, -k:, :,
                                                                                            1]
            train_batch_disp_45d[:, :, :k, k + param.disp] = train_batch_45d[:, :, :k, 0] - train_batch_45d[:, :, -k:,
                                                                                            1]
            train_batch_disp_m45d[:, :, :k, k + param.disp] = train_batch_m45d[:, :, :k, 0] - train_batch_m45d[:, :,
                                                                                              -k:, 1]
        elif k > 0:
            train_batch_disp_0d[:, k:, :, k + param.disp] = train_batch_0d[:, k:, :, 0] - train_batch_0d[:, :-k, :, 1]
            train_batch_disp_90d[:, k:, :, k + param.disp] = train_batch_90d[:, k:, :, 0] - train_batch_90d[:, :-k, :,
                                                                                            1]
            train_batch_disp_45d[:, :, k:, k + param.disp] = train_batch_45d[:, :, k:, 0] - train_batch_45d[:, :, :-k,
                                                                                            1]
            train_batch_disp_m45d[:, :, k:, k + param.disp] = train_batch_m45d[:, :, k:, 0] - train_batch_m45d[:, :,
                                                                                              :-k, 1]
        elif k == 0:
            train_batch_disp_0d[:, :, :, k + param.disp] = train_batch_0d[:, :, :, 0] - train_batch_0d[:, :, :, 1]
            train_batch_disp_90d[:, :, :, k + param.disp] = train_batch_90d[:, :, :, 0] - train_batch_90d[:, :, :, 1]
            train_batch_disp_45d[:, :, :, k + param.disp] = train_batch_45d[:, :, :, 0] - train_batch_45d[:, :, :, 1]
            train_batch_disp_m45d[:, :, :, k + param.disp] = train_batch_m45d[:, :, :, 0] - train_batch_m45d[:, :, :, 1]

    return train_batch_disp_0d, train_batch_disp_90d, train_batch_disp_45d, train_batch_disp_m45d


def generate_train_data(raw_data_90d,raw_data_0d, raw_data_45d, raw_data_m45d, raw_label):
    """

    :param raw_data_0d: ndarray(512,512,3,9,n_scences)
    :param raw_data_90d:
    :param raw_data_45d:
    :param raw_data_m45d:
    :param raw_label: ndarray(512,512,9,n_scences)
    :return: train_batch_0d(batch_size,23~,23~,9)
            train_batch_label(batch_size,1~,1~)
    """
    train_batch_0d = np.zeros((param.input_size, param.input_size, param.each_row_pic_num, param.batch_size), dtype=np.float32)
    train_batch_90d = np.zeros((param.input_size, param.input_size, param.each_row_pic_num, param.batch_size), dtype=np.float32)
    train_batch_45d = np.zeros((param.input_size, param.input_size, param.each_row_pic_num, param.batch_size), dtype=np.float32)
    train_batch_m45d = np.zeros((param.input_size, param.input_size, param.each_row_pic_num, param.batch_size), dtype=np.float32)
    train_batch_label = np.zeros((param.output_size, param.output_size, param.batch_size), dtype=np.float32)

    all_scences = np.array(list(range(len(param.trainset_dirs))))

    for i in range(param.batch_size):
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

            crop_0d = random_gray_resized_crop(x_start, y_start, scale, scence_id, raw_data_0d, R, G, B)

            # sum_diff
            # center_img = (1 / 255) * crop_0d[:, :, 4]
            center_img = (1 / 255) * crop_0d[:, :, param.each_row_pic_num//2]
            sum_diff = np.sum(
                np.abs(center_img - np.squeeze(center_img[int(0.5 * param.input_size), int(0.5 * param.input_size)])))

            if sum_diff < 0.01 * param.input_size * param.input_size:
                continue

            else:
                crop_90d = random_gray_resized_crop(x_start, y_start, scale, scence_id, raw_data_90d, R, G, B)
                crop_45d = random_gray_resized_crop(x_start, y_start, scale, scence_id, raw_data_45d, R, G, B)
                crop_m45d = random_gray_resized_crop(x_start, y_start, scale, scence_id, raw_data_m45d, R, G, B)
                crop_label = random_resized_crop(x_start, y_start, scale, scence_id, raw_label)

                train_batch_0d[:, :, :, i] = crop_0d
                train_batch_90d[:, :, :, i] = crop_90d
                train_batch_45d[:, :, :, i] = crop_45d
                train_batch_m45d[:, :, :, i] = crop_m45d
                train_batch_label[:, :, i] = crop_label
                break

    train_batch_0d = np.transpose(train_batch_0d, (3, 0, 1, 2)) / 255
    train_batch_90d = np.transpose(train_batch_90d, (3, 0, 1, 2)) / 255
    train_batch_45d = np.transpose(train_batch_45d, (3, 0, 1, 2)) / 255
    train_batch_m45d = np.transpose(train_batch_m45d, (3, 0, 1, 2)) / 255
    train_batch_label = np.transpose(train_batch_label, (2, 0, 1))

    # train_batch_disp_0d, train_batch_disp_90d, train_batch_disp_45d, train_batch_disp_m45d = \
    #     build_cost_volume(train_batch_0d, train_batch_90d, train_batch_45d, train_batch_m45d)
    """
    # TODO: 构架cost volume
    '''
    0d  -> 4 40      90d -> 76 40
    45d -> 36 40    m45d -> 44 40
    '''
    train_batch_disp_0d = np.zeros((param.batch_size, param.input_size, param.input_size, param.disp * 2 + 1), dtype=np.float32)
    train_batch_disp_90d = np.zeros((param.batch_size, param.input_size, param.input_size, param.disp * 2 + 1), dtype=np.float32)
    train_batch_disp_45d = np.zeros((param.batch_size, param.input_size, param.input_size, param.disp * 2 + 1), dtype=np.float32)
    train_batch_disp_m45d = np.zeros((param.batch_size, param.input_size, param.input_size, param.disp * 2 + 1), dtype=np.float32)

    # #测试图像分割
    # vis1 = np.zeros(shape=(param.input_size, param.input_size, 1), dtype=np.float32)
    # vis2 = np.zeros(shape=(param.input_size, param.input_size, 1), dtype=np.float32)
    # vis1[:-15, :, 0] = train_batch_0d[0, :-15, :, 0]
    # vis2[:, :, 0] = train_batch_0d[0, :, :, 0]
    # imageio.imwrite(r'1.jpg', vis1)
    # imageio.imwrite(r'2.jpg', vis2)

    # 先Hight后width
    for k in range(-param.disp, param.disp+1):
        if k < 0:
            train_batch_disp_0d[:, :k, :, k + param.disp] = train_batch_0d[:, :k, :, 0] - train_batch_0d[:, -k:, :, 1]
            train_batch_disp_90d[:, :k, :, k + param.disp] = train_batch_90d[:, :k, :, 0] - train_batch_90d[:, -k:, :, 1]
            train_batch_disp_45d[:, :, :k, k + param.disp] = train_batch_45d[:, :, :k, 0] - train_batch_45d[:, :, -k:, 1]
            train_batch_disp_m45d[:, :, :k, k + param.disp] = train_batch_m45d[:, :, :k, 0] - train_batch_m45d[:, :, -k:, 1]
        elif k > 0:
            train_batch_disp_0d[:, k:, :, k + param.disp] = train_batch_0d[:, k:, :, 0] - train_batch_0d[:, :-k, :, 1]
            train_batch_disp_90d[:, k:, :, k + param.disp] = train_batch_90d[:, k:, :, 0] - train_batch_90d[:, :-k, :, 1]
            train_batch_disp_45d[:, :, k:, k + param.disp] = train_batch_45d[:, :, k:, 0] - train_batch_45d[:, :, :-k, 1]
            train_batch_disp_m45d[:, :, k:, k + param.disp] = train_batch_m45d[:, :, k:, 0] - train_batch_m45d[:, :, :-k, 1]
        elif k == 0:
            train_batch_disp_0d[:, :, :, k + param.disp] = train_batch_0d[:, :, :, 0] - train_batch_0d[:, :, :, 1]
            train_batch_disp_90d[:, :, :, k + param.disp] = train_batch_90d[:, :, :, 0] - train_batch_90d[:, :, :, 1]
            train_batch_disp_45d[:, :, :, k + param.disp] = train_batch_45d[:, :, :, 0] - train_batch_45d[:, :, :, 1]
            train_batch_disp_m45d[:, :, :, k + param.disp] = train_batch_m45d[:, :, :, 0] - train_batch_m45d[:, :, :, 1]
    """

    return train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label
    # return train_batch_disp_90d, train_batch_disp_0d, train_batch_disp_45d, train_batch_disp_m45d, train_batch_label


def aug_operation(train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label):
    for batch_i in range(param.batch_size):
        gray_rand = 0.4 * np.random.rand() + 0.8

        train_batch_90d[batch_i, :, :, :] = pow(train_batch_90d[batch_i, :, :, :], gray_rand)
        train_batch_0d[batch_i, :, :, :] = pow(train_batch_0d[batch_i, :, :, :], gray_rand)
        train_batch_45d[batch_i, :, :, :] = pow(train_batch_45d[batch_i, :, :, :], gray_rand)
        train_batch_m45d[batch_i, :, :, :] = pow(train_batch_m45d[batch_i, :, :, :], gray_rand)

        rotation_or_transp_rand = np.random.randint(0, 5)

        if rotation_or_transp_rand == 4:
            train_batch_90d_tmp6 = np.copy(np.transpose(np.squeeze(train_batch_90d[batch_i, :, :, :]), (1, 0, 2)))
            train_batch_0d_tmp6 = np.copy(np.transpose(np.squeeze(train_batch_0d[batch_i, :, :, :]), (1, 0, 2)))
            train_batch_45d_tmp6 = np.copy(np.transpose(np.squeeze(train_batch_45d[batch_i, :, :, :]), (1, 0, 2)))
            train_batch_m45d_tmp6 = np.copy(np.transpose(np.squeeze(train_batch_m45d[batch_i, :, :, :]), (1, 0, 2)))

            train_batch_0d[batch_i, :, :, :] = np.copy(train_batch_90d_tmp6[:, :, ::-1])
            train_batch_90d[batch_i, :, :, :] = np.copy(train_batch_0d_tmp6[:, :, ::-1])
            train_batch_45d[batch_i, :, :, :] = np.copy(train_batch_45d_tmp6[:, :, ::-1])
            train_batch_m45d[batch_i, :, :, :] = np.copy(train_batch_m45d_tmp6)  # [:,:,::-1])
            train_batch_label[batch_i, :, :] = np.copy(
                np.transpose(train_batch_label[batch_i, :, :], (1, 0)))

        if (rotation_or_transp_rand == 1):  # 90d

            train_batch_90d_tmp3 = np.copy(np.rot90(train_batch_90d[batch_i, :, :, :], 1, (0, 1)))
            train_batch_0d_tmp3 = np.copy(np.rot90(train_batch_0d[batch_i, :, :, :], 1, (0, 1)))
            train_batch_45d_tmp3 = np.copy(np.rot90(train_batch_45d[batch_i, :, :, :], 1, (0, 1)))
            train_batch_m45d_tmp3 = np.copy(np.rot90(train_batch_m45d[batch_i, :, :, :], 1, (0, 1)))

            train_batch_90d[batch_i, :, :, :] = train_batch_0d_tmp3
            train_batch_45d[batch_i, :, :, :] = train_batch_m45d_tmp3
            train_batch_0d[batch_i, :, :, :] = train_batch_90d_tmp3[:, :, ::-1]
            train_batch_m45d[batch_i, :, :, :] = train_batch_45d_tmp3[:, :, ::-1]

            train_batch_label[batch_i, :, :] = np.copy(
                np.rot90(train_batch_label[batch_i, :, :], 1, (0, 1)))

        if (rotation_or_transp_rand == 2):  # 180d

            train_batch_90d_tmp4 = np.copy(np.rot90(train_batch_90d[batch_i, :, :, :], 2, (0, 1)))
            train_batch_0d_tmp4 = np.copy(np.rot90(train_batch_0d[batch_i, :, :, :], 2, (0, 1)))
            train_batch_45d_tmp4 = np.copy(np.rot90(train_batch_45d[batch_i, :, :, :], 2, (0, 1)))
            train_batch_m45d_tmp4 = np.copy(np.rot90(train_batch_m45d[batch_i, :, :, :], 2, (0, 1)))

            train_batch_90d[batch_i, :, :, :] = train_batch_90d_tmp4[:, :, ::-1]
            train_batch_0d[batch_i, :, :, :] = train_batch_0d_tmp4[:, :, ::-1]
            train_batch_45d[batch_i, :, :, :] = train_batch_45d_tmp4[:, :, ::-1]
            train_batch_m45d[batch_i, :, :, :] = train_batch_m45d_tmp4[:, :, ::-1]

            train_batch_label[batch_i, :, :] = np.copy(
                np.rot90(train_batch_label[batch_i, :, :], 2, (0, 1)))

        if (rotation_or_transp_rand == 3):  # 270d

            train_batch_90d_tmp5 = np.copy(np.rot90(train_batch_90d[batch_i, :, :, :], 3, (0, 1)))
            train_batch_0d_tmp5 = np.copy(np.rot90(train_batch_0d[batch_i, :, :, :], 3, (0, 1)))
            train_batch_45d_tmp5 = np.copy(np.rot90(train_batch_45d[batch_i, :, :, :], 3, (0, 1)))
            train_batch_m45d_tmp5 = np.copy(np.rot90(train_batch_m45d[batch_i, :, :, :], 3, (0, 1)))

            train_batch_90d[batch_i, :, :, :] = train_batch_0d_tmp5[:, :, ::-1]
            train_batch_0d[batch_i, :, :, :] = train_batch_90d_tmp5
            train_batch_45d[batch_i, :, :, :] = train_batch_m45d_tmp5[:, :, ::-1]
            train_batch_m45d[batch_i, :, :, :] = train_batch_45d_tmp5

            train_batch_label[batch_i, :, :] = np.copy(
                np.rot90(train_batch_label[batch_i, :, :], 3, (0, 1)))

    return train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label


if __name__ == '__main__':
    raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label = read_data(param.trainset_dirs, param.idx_90d,
                                                                                  param.idx_0d, param.idx_45d,
                                                                                  param.idx_m45d, param.input_img_size)

    train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label = generate_train_data(
        raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label)