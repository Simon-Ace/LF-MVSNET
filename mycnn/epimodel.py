from keras.layers import Input, Activation, Conv2D, concatenate, BatchNormalization, Reshape, LeakyReLU, add
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
import param
from keras.utils import plot_model
from keras import backend as K


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


def get_model(filters_count, conv_depth, learning_rate, input_shape=(512, 512, 9)):
    # input
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
    merged = concatenate([stream_ver,stream_hor, stream_45d, stream_m45d], name='merged')

    # layers part2: conv-relu-bn-conv-relu
    merged = layersP2_merged(input_shape=(input_shape[0]-6, input_shape[1]-6, int(filters_count) * 4),
                             filters_count=int(filters_count) * 4,
                             conv_depth=conv_depth)(merged)

    # output
    output = layersP3_output(input_shape=(input_shape[0]-18, input_shape[1]-18, int(filters_count) * 4),
                             filters_count=int(filters_count) * 4)(merged)

    mymodel = Model(inputs=[input_90d,input_0d, input_45d, input_m45d], outputs=[output])

    optimizer = RMSprop(lr=learning_rate)
    mymodel.compile(optimizer=optimizer, loss='mae')
    mymodel.summary()

    return mymodel



# ======================================== MVS_MODEL ============================================
# Resdidual Block
def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    for i in range(2):
        y = Conv2D(nb_channels, kernel_size=(2, 2), strides=_strides, padding='same')(y)
        y = BatchNormalization()(y)
        y = LeakyReLU(alpha=0.2)(y)

    """
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    """

    y = add([shortcut, y])
    #y = LeakyReLU()(y)
    return y

def FeatureExtraction(input_shape, filters_count, feature_depth):
    '''

    :param input_shape:
    :param filters_count:
    :param feature_depth:
    :return:
    '''
    #seq = Sequential()
    input = Input(shape=input_shape)
    X = Conv2D(filters_count, kernel_size=(2, 2), strides=(1,1), padding='same')(input)
    for i in range(feature_depth):
        X = residual_block(X, filters_count)
    return Model(inputs=input, outputs=X)



def FeatureExtraction_ori(input_shape, filters_count, feature_depth):
    """

    :param input_shape:
    :param filters_count:
    :param feature_depth:
    :return:
    """
    pass




def get_mvs_model(feature_filters_num, feature_depth, cost_filter_depth, learning_rate, input_pic_num, input_shape=(512, 512, 1)):
    '''
    :param feature_filters_num: 特征提取部分卷积核数量
    :param feature_depth: 特征提取部分网络层数
    :param cost_filter_depth: cost volume filter部分网络层数
    :param learning_rate: 学习速率
    :param input_pic_num: 输入图片数量
    :param input_shape: 输入尺寸 eg. (512, 512, 1)
    :return:

    网络整体结构：
    Siamese Network(权值共享) -> 中心视角对每个图的cost volume -> cost volume求均值 -> cost volume filtering -> WTA -> Refine
    '''
    # TODO: 输入层
    input_list = []
    for i in range(input_pic_num):
        input_pic = Input(shape=input_shape, name='input_d{}'.format(i))
        input_list.append(input_pic)

    #TODO: Siamese Network
    feature_list = []

    '''
    # ResBlock 孪生网络
    fea_ext_model = FeatureExtraction(input_shape, feature_filters_num, feature_depth)
    for i in range(input_pic_num):
        X = fea_ext_model(input_list[i])
        feature_list.append(X)
    '''

    # 非孪生原始网络
    for i in range(input_pic_num):
        X = layersP1_multistream(input_shape, feature_filters_num)(input_list[i])
        feature_list.append(X)

    # 孪生原始网络
    # layer1_seq = layersP1_multistream(input_shape, feature_filters_num)
    # for i in range(input_pic_num):
    #     X = layer1_seq(input_list[i])
    #     feature_list.append(X)

    # TODO: cost volume构建
    disp = 3
    cost_volume = K.zeros(shape=(X.shape[0], X.shape[1], disp, X.shape[2], X.shape[3]))


    merged = concatenate(feature_list, name='merged')


    # 瞎写的:把原来的抄过来
    merged = layersP2_merged(input_shape=(input_shape[0], input_shape[1], int(feature_filters_num) * 5),
                             filters_count=int(feature_filters_num) * 5,
                             conv_depth=cost_filter_depth)(merged)

    # output
    output = layersP3_output(input_shape=(input_shape[0] - 14, input_shape[1] - 14, int(feature_filters_num) * 5),
                             filters_count=int(feature_filters_num) * 5)(merged)

    mymodel = Model(inputs=input_list, outputs=[output])

    optimizer = RMSprop(lr=learning_rate)
    mymodel.compile(optimizer=optimizer, loss='mae')
    mymodel.summary()

    return mymodel





    #TODO: mean cost volume

    #TODO: cost colume filtering

    #TODO: WTA

    #TODO: Refinement

    '''
    model = Model(inputs=input_list, outputs=merged)
    print(model.summary())

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    '''


    return None

if __name__ == '__main__':
    model_filter_nums = 32      #卷积核数量
    model_feature_depth = 3     #特征提取部分，网络层数
    model_conv_depth = 7        #卷积层数（后面）
    model_learning_rate = 1e-4  #LR
    input_size = 128             #图像输入大小

    import os
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    mymodel = get_mvs_model(model_filter_nums, model_feature_depth, model_conv_depth, model_learning_rate,
                            input_pic_num=5, input_shape=(input_size, input_size, 1))
    plot_model(mymodel, to_file='model_all_4.png')

