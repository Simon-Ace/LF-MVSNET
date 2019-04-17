from keras.layers import Input, Activation, Conv2D, concatenate, BatchNormalization,Reshape, Concatenate, Lambda
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.activations import softmax
import keras.backend as K
import tensorflow as tf
import numpy as np
from param import *



# 4 stream input sequence
def layersP1_multistream(input_shape, filters_count):
    seq = Sequential()
    for i in range(3):
        seq.add(Conv2D(int(filters_count), (2, 2),
                       input_shape=input_shape,
                       padding='valid',
                       name='seq1_conv1_%d' % (i)))
        seq.add(Activation('relu', name='seq1_relu1_%d' % i))
        seq.add(Conv2D(int(filters_count), (2, 2),
                       padding='valid',
                       name='seq1_conv2_%d' % (i)))
        seq.add(BatchNormalization(axis=-1, name='seq1_BN_%d' % i))
        seq.add(Activation('relu', name='seq1_relu2_%d' % i))

    seq.add(Reshape((input_shape[0] - 6, input_shape[1] - 6, int(filters_count))))
    return seq


# merged_layer: conv-relu-conv-relu
def layersP2_merged(input_shape, filters_count, conv_depth):
    seq = Sequential()
    for i in range(conv_depth):
        seq.add(Conv2D(filters_count, (2, 2),
                       padding='valid',
                       input_shape=input_shape,
                       name='seq2_conv1_%d' % (i)))
        seq.add(Activation('relu', name='seq2_relu1_%d' % i))
        seq.add(Conv2D(filters_count, (2, 2),
                       padding='valid',
                       input_shape=input_shape,
                       name='seq2_conv2_%d' % (i)))
        seq.add(BatchNormalization(axis=-1, name='seq2_BN_%d' % i))
        seq.add(Activation('relu', name='seq2_relu2_%d' % i))
    return seq


# output layer
def layersP3_output(input_shape, filters_count):
    seq = Sequential()
    seq.add(Conv2D(filters_count, (2, 2),
                   padding='valid',
                   input_shape=input_shape,
                   activation='relu',
                   name='seq3_conv1_0'))
    seq.add(Conv2D(1, (2, 2), padding='valid', name='seq3_last'))
    return seq


def get_model(filters_count, conv_depth, learning_rate, input_shape=(512, 512, 9), is_train=True):
    # input
    input_90d = Input(shape=input_shape, name='input_90d')
    input_0d = Input(shape=input_shape, name='input_0d')
    input_45d = Input(shape=input_shape, name='input_45d')
    input_m45d = Input(shape=input_shape, name='input_m45d')

    # 4 Stream layer
    # 0415 - 修改分叉部分通道数
    filters_count = input_90d.shape[-1]
    stream_ver = layersP1_multistream(input_shape, int(filters_count))(input_90d)
    stream_hor = layersP1_multistream(input_shape, int(filters_count))(input_0d)
    stream_45d = layersP1_multistream(input_shape, int(filters_count))(input_45d)
    stream_m45d = layersP1_multistream(input_shape, int(filters_count))(input_m45d)

    # TODO 0415 - soft argmin
    # soft_stream_ver = tf.nn.softmax(tf.scalar_mul(-1, stream_ver), axis=3)
    # soft_stream_hor = tf.nn.softmax(tf.scalar_mul(-1, stream_hor), axis=3)
    # soft_stream_45d = tf.nn.softmax(tf.scalar_mul(-1, stream_45d), axis=3)
    # soft_stream_m45d = tf.nn.softmax(tf.scalar_mul(-1, stream_m45d), axis=3)

    # stream_ver = Lambda(lambda x: tf.scalar_mul(-1, x))(stream_ver)
    # stream_hor = Lambda(lambda x: tf.scalar_mul(-1, x))(stream_hor)
    # stream_45d = Lambda(lambda x: tf.scalar_mul(-1, x))(stream_45d)
    # stream_m45d = Lambda(lambda x: tf.scalar_mul(-1, x))(stream_m45d)
    #
    # soft_stream_ver = Lambda(lambda x: K.softmax(x, axis=3))(stream_ver)
    # soft_stream_hor = Lambda(lambda x: K.softmax(x, axis=3))(stream_hor)
    # soft_stream_45d = Lambda(lambda x: K.softmax(x, axis=3))(stream_45d)
    # soft_stream_m45d = Lambda(lambda x: K.softmax(x, axis=3))(stream_m45d)

    soft_stream_ver = Lambda(lambda x: K.softmax(tf.scalar_mul(-1, x), axis=3))(stream_ver)
    soft_stream_hor = Lambda(lambda x: K.softmax(tf.scalar_mul(-1, x), axis=3))(stream_hor)
    soft_stream_45d = Lambda(lambda x: K.softmax(tf.scalar_mul(-1, x), axis=3))(stream_45d)
    soft_stream_m45d = Lambda(lambda x: K.softmax(tf.scalar_mul(-1, x), axis=3))(stream_m45d)


    disp_init = [i for i in range(param.disp+1)]
    disp_re = disp_init[-1:0:-1]
    disp_re.extend(disp_init)

    disp_list = np.asarray(disp_re)
    first_shape = param.batch_size if is_train else 1
    disp_list = np.tile(disp_list, (first_shape, stream_ver.shape[1], stream_ver.shape[2], 1))
    # disp = variable(np.reshape(np.asarray(disp_re), [1, 1, 1, param.disp*2+1]))
    disp_list = K.variable(disp_list)

    # softarg_mid_ver = Lambda(lambda x: x * disp_list)(soft_stream_ver)
    # softarg_mid_hor = Lambda(lambda x: x * disp_list)(soft_stream_hor)
    # softarg_mid_45d = Lambda(lambda x: x * disp_list)(soft_stream_45d)
    # softarg_mid_m45d = Lambda(lambda x: x * disp_list)(soft_stream_m45d)
    #
    # pred_ver = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(softarg_mid_ver)
    # pred_hor = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(softarg_mid_hor)
    # pred_45d = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(softarg_mid_45d)
    # pred_m45d = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(softarg_mid_m45d)


    pred_ver = Lambda(lambda x: K.sum(x * disp_list, axis=3, keepdims=True))(soft_stream_ver)
    pred_hor = Lambda(lambda x: K.sum(x * disp_list, axis=3, keepdims=True))(soft_stream_hor)
    pred_45d = Lambda(lambda x: K.sum(x * disp_list, axis=3, keepdims=True))(soft_stream_45d)
    pred_m45d = Lambda(lambda x: K.sum(x * disp_list, axis=3, keepdims=True))(soft_stream_m45d)

    '''
    pred_ver = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(soft_stream_ver * disp)
    pred_hor = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(soft_stream_hor * disp)
    pred_45d = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(soft_stream_45d * disp)
    pred_m45d = Lambda(lambda x: K.sum(x, axis=3, keepdims=True))(soft_stream_m45d * disp)

    # pred_ver = K.sum(soft_stream_ver * disp, axis=3, keepdims=True)
    # pred_hor = K.sum(soft_stream_hor * disp, axis=3, keepdims=True)
    # pred_45d = K.sum(soft_stream_45d * disp, axis=3, keepdims=True)
    # pred_m45d = K.sum(soft_stream_m45d * disp, axis=3, keepdims=True)
    '''


    # merge streams
    # merged = concatenate([stream_ver, stream_hor, stream_45d, stream_m45d], name='merged')
    merged = concatenate([pred_ver, pred_hor, pred_45d, pred_m45d], name='merged')
    merged = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(merged)

    # layers part2: conv-relu-bn-conv-relu
    # merged = layersP2_merged(input_shape=(input_shape[0]-6, input_shape[1]-6, int(filters_count) * 4),
    #                          filters_count=int(filters_count) * 4,
    #                          conv_depth=conv_depth)(merged)
    #
    # # output
    # output = layersP3_output(input_shape=(input_shape[0]-18, input_shape[1]-18, int(filters_count) * 4),
    #                          filters_count=int(filters_count) * 4)(merged)

    # layers part2: conv-relu-bn-conv-relu
    merged = layersP2_merged(input_shape=(input_shape[0] - 6, input_shape[1] - 6, 1),
                             filters_count=32,
                             conv_depth=conv_depth)(merged)

    # output
    output = layersP3_output(input_shape=(input_shape[0] - 18, input_shape[1] - 18, 32),
                             filters_count=32)(merged)

    mymodel = Model(inputs=[input_90d, input_0d, input_45d, input_m45d], outputs=[output])

    optimizer = RMSprop(lr=learning_rate)
    mymodel.compile(optimizer=optimizer, loss='mae')
    #mymodel.summary()

    return mymodel

def get_model_test():

    input_shape = (512,512,9)
    filters_count = 70
    conv_depth = 7
    input_90d = Input(shape=input_shape, name='input_90d')
    input_0d = Input(shape=input_shape, name='input_0d')
    input_45d = Input(shape=input_shape, name='input_45d')
    input_m45d = Input(shape=input_shape, name='input_m45d')

    # 4 Stream layer
    stream_ver = layersP1_multistream(input_shape, int(filters_count))(input_90d)
    stream_hor = layersP1_multistream(input_shape, int(filters_count))(input_0d)
    stream_45d = layersP1_multistream(input_shape, int(filters_count))(input_45d)
    stream_m45d = layersP1_multistream(input_shape, int(filters_count))(input_m45d)

    # merge streams
    merged = concatenate([stream_ver, stream_hor, stream_45d, stream_m45d], name='merged')

    # layers part2: conv-relu-bn-conv-relu
    merged = layersP2_merged(input_shape=(input_shape[0] - 6, input_shape[1] - 6, int(filters_count) * 4),
                             filters_count=int(filters_count) * 4,
                             conv_depth=conv_depth)(merged)

    # output
    output = layersP3_output(input_shape=(input_shape[0]-20, input_shape[1]-20, int(filters_count) * 4),
                             filters_count=int(filters_count) * 4)(merged)

    mymodel = Model(inputs=[input_90d, input_0d, input_45d, input_m45d], outputs=[output])

    optimizer = RMSprop(lr=1e-3)
    mymodel.compile(optimizer=optimizer, loss='mae')
    mymodel.summary()

    return mymodel

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # model = get_model_test()
    model = get_model(param.model_filter_nums, param.model_conv_depth, param.model_learning_rate,
                      input_shape=(param.input_size, param.input_size, param.disp * 2 + 1))
    model.summary()