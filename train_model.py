import os
from param import *
import datetime
import numpy as np
from data_process.utils import *
from mycnn.epimodel import get_model
import re
import time
import utils
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


import threading


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def epi_generator(raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label):
    while 1:
        # t0 = time.time()
        train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label = generate_train_data(
            raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label)
        # print("COST TIME: %f(s)" % (time.time() - t0))

        # # 数据增强
        # train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label = aug_operation(
        #     train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label)

        # 数据增强
        train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label = mycost_aug_operation(
            train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d, train_batch_label)

        # cost volume
        train_batch_disp_0d, train_batch_disp_90d, train_batch_disp_45d, train_batch_disp_m45d = \
            build_cost_volume(train_batch_0d, train_batch_90d, train_batch_45d, train_batch_m45d)

        # yield (
        # [train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d], train_batch_label[:, :, :, np.newaxis])
        yield ([train_batch_disp_90d, train_batch_disp_0d, train_batch_disp_45d, train_batch_disp_m45d],
                    train_batch_label[:, :, :, np.newaxis])


if __name__ == '__main__':
    # GPU setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # dir check
    if not os.path.exists(param.iter_dir):
        os.makedirs(param.iter_dir)
    if not os.path.exists(param.checkpoint_dir):
        os.makedirs(param.checkpoint_dir)

    # define 2 model for training and validation
    # 写死了9x9的图？
    # model = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
    #                   input_shape=(param.input_size, param.input_size, param.each_row_pic_num))
    # model_512 = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
    #                       input_shape=(param.input_img_size, param.input_img_size, param.each_row_pic_num))

    model = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
                      input_shape=(param.input_size, param.input_size, param.disp * 2 + 1))
    model.summary()
    model_512 = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
                          input_shape=(param.input_img_size, param.input_img_size, param.disp * 2 + 1))

    initial_epoch = 0
    best_result = 100.0  # best result for past training
    if param.is_continue:
        # load the latest checkpoints and initial_epoch
        ckps = os.listdir(param.checkpoint_dir)
        ckps.sort()
        latest_ckp = ckps[-1]
        model.load_weights(os.path.join(param.checkpoint_dir, latest_ckp))
        initial_epoch = int(re.findall(r'\d+', latest_ckp)[0])
        best_result = float(latest_ckp.split('_')[-1][2:-5])

    # load the raw train data  label是GT差异图
    raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label = read_data(param.trainset_dirs, param.idx_90d,
                                                                                  param.idx_0d, param.idx_45d,
                                                                                  param.idx_m45d, param.input_img_size)

    # vis2 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    # vis2[:,:,:] = raw_data_90d[:, :, :, 0, 0]
    # imageio.imwrite(r'1.jpg', vis2)


    # 512x512x3x9 -> 512x512x9
    # 后面加了差异图转换cost volume √
    train512_data_90d, train512_data_0d, train512_data_45d, train512_data_m45d = stack_data(raw_data_90d, raw_data_0d,
                                                                                            raw_data_45d, raw_data_m45d)

    # generator
    mygenerator = epi_generator(raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label)

    # TensorBoard
    # from keras.callbacks import TensorBoard
    # tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
    #                          histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                          #batch_size=32,     # 用多大量的数据计算直方图
    #                          write_graph=True,  # 是否存储网络结构图
    #                          write_grads=True,  # 是否可视化梯度直方图
    #                          write_images=True,  # 是否可视化参数
    #                          embeddings_freq=0,
    #                          embeddings_layer_names=None,
    #                          embeddings_metadata=None)

    while 1:
        history = model.fit_generator(mygenerator, steps_per_epoch=param.steps_per_epoch, epochs=initial_epoch + 1,
                            initial_epoch=initial_epoch,
                            verbose=1, workers=param.worker_num, max_queue_size=10, use_multiprocessing=False)
        initial_epoch += 1

        weights = model.get_weights()
        model_512.set_weights(weights)
        train512_output = model_512.predict(
            [train512_data_90d, train512_data_0d, train512_data_45d, train512_data_m45d], batch_size=1)

        train_diff, train_bp = utils.get_diff_badpixel(train512_output, raw_label, initial_epoch, param.iter_dir)
        training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
        training_bad_pixel_ratio = 100 * np.average(train_bp)

        print('-------------------------iter%04d_mse%.3f_bp%.2f.hdf5' % (
        initial_epoch, training_mean_squared_error_x100, training_bad_pixel_ratio))
        # checkpoint to save
        ckp_file = param.checkpoint_dir + 'iter%04d_mse%.3f_bp%.2f.hdf5' % (
        initial_epoch, training_mean_squared_error_x100, training_bad_pixel_ratio)

        # logfile
        with open(param.logfile_dir, 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + \
                    ' iter%04d_mse-%5.3f_bp-%4.2f_trainLoss-%.4f' % (
                    initial_epoch, training_mean_squared_error_x100, training_bad_pixel_ratio, history.history['loss'][0]) \
                    + '\n')

        if (training_bad_pixel_ratio < best_result):
            best_result = training_bad_pixel_ratio
            model.save_weights(ckp_file)
            print("-----------saved------------")
