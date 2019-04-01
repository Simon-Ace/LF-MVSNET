from keras.layers import Input, Activation, Conv2D, concatenate, BatchNormalization,Reshape
from keras.optimizers import RMSprop
from keras.models import Model, Sequential


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
