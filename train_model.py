import os
from param import *
import datetime
import numpy as np
from data_process.utils import *
from data_process.utils_my import *
from mycnn.epimodel import get_model, get_mvs_model
import re
import time
import utils

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
        yield (
        [train_batch_90d, train_batch_0d, train_batch_45d, train_batch_m45d], train_batch_label[:, :, :, np.newaxis])


@threadsafe_generator
def mvs_generator(raw_data_list, raw_label):
    while 1:
        train_batch_list, train_batch_label = generate_any_num_train_data(raw_data_list, raw_label)

        yield (train_batch_list, train_batch_label[:, :, :, np.newaxis])


if __name__ == '__main__':
    # GPU setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # dir check
    if not os.path.exists(param.iter_dir):
        os.makedirs(param.iter_dir)
    if not os.path.exists(param.checkpoint_dir):
        os.makedirs(param.checkpoint_dir)

    # define 2 model for training and validation
    # 写死了9x9的图？
    '''
    model = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
                      input_shape=(param.input_size, param.input_size, 9))
    model_512 = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
                          input_shape=(param.input_img_size, param.input_img_size, 9))
    '''
    # TODO: input_size记得改
    # model = get_mvs_model(32, 8, param.model_learning_rate, input_pic_num=5,
    #                       input_shape=(param.input_size, param.input_size, 1))
    # model_512 = get_mvs_model(32, 8, param.model_learning_rate, input_pic_num=5,
    #                           input_shape=(param.input_img_size, param.input_img_size, 1))
    model = get_mvs_model(param.feature_filters_num, param.featrue_depth, param.cost_filter_depth,
                          param.model_learning_rate, param.input_pic_num,
                          input_shape=(param.input_size, param.input_size, 1))
    model_512 = get_mvs_model(param.feature_filters_num, param.featrue_depth, param.cost_filter_depth,
                          param.model_learning_rate, param.input_pic_num,
                          input_shape=(param.input_img_size, param.input_img_size, 1))

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
    # raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label = read_data(param.trainset_dirs, param.idx_90d,
    #                                                                               param.idx_0d, param.idx_45d,
    #                                                                               param.idx_m45d, param.input_img_size)
    raw_data_list, raw_label = read_any_num_data(param.trainset_dirs, param.input_img_size, param.idx_5pic)

    # 512x512x3x9 -> 512x512x9
    # train512_data_90d, train512_data_0d, train512_data_45d, train512_data_m45d = stack_data(raw_data_90d, raw_data_0d,
    #                                                                                         raw_data_45d, raw_data_m45d)
    train512_data_list = stack_any_num_data(raw_data_list)


    # generator
    # mygenerator = epi_generator(raw_data_90d, raw_data_0d, raw_data_45d, raw_data_m45d, raw_label)
    mygenerator = mvs_generator(raw_data_list, raw_label)

    while 1:
        model.fit_generator(mygenerator, steps_per_epoch=param.steps_per_epoch, epochs=initial_epoch + 1,
                            initial_epoch=initial_epoch,
                            verbose=1, workers=param.worker_num, max_queue_size=10, use_multiprocessing=False)
        initial_epoch += 1

        weights = model.get_weights()
        model_512.set_weights(weights)
        # train512_output = model_512.predict(
        #     [train512_data_90d, train512_data_0d, train512_data_45d, train512_data_m45d], batch_size=1)

        train512_output = model_512.predict(train512_data_list, batch_size=1)

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
                    'iter%04d_mse%.3f_bp%.2f.hdf5' % (
                    initial_epoch, training_mean_squared_error_x100, training_bad_pixel_ratio) \
                    + '\n')

        if (training_bad_pixel_ratio < best_result):
            best_result = training_bad_pixel_ratio
            model.save_weights(ckp_file)
            print("-----------saved------------")
    


