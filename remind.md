## 0 添加自动化测试！！！

## 1 细节部分
- 添加 Keras TensorBoard
- callbacks里面加lr自动缩减
- batch可以加大一点，GPU利用率不高


## 2 模型结构

## 3 工作记录
**20190402**  
- [x] `utils_my.py`  stack_data()没写完

- [x] `get_mvs_model`这东西某个地方的参数不对，不把前两个参数写成32和8会报错，查一下哪的问题

**20190403**
- [x] `random_gray_resized_crop` 会把维度从(512, 512, 3, 9, 13)降至(25, 25, 9)，  
我改过的就会从(512, 512, 3, 13)降至(25, 25)少了一个维度，得加回来  
加个维度->  
`X_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)`

- [x] `generate_any_num_train_data()`写了一半，输出部分还没改好

**20190404**
- [x] `generate_any_num_train_data()`改好了
- [x] param.py `param.output_size = param.input_size - 18` 数值不能写死，根据卷机网络层数进行计算
- [x] 根目录utils.py `train_output482 = train_output[:,4:-4,4:-4]` 也得改，不能写死
- [x] 测试一下epimodel，改成了孪生网络

remind:  
(改回去了)前面ResBlock的padding改成vaild，好像不能改=。=，图片维度不一样了，没法加  
StereoNet前面也是ResNet怎么做的？？【加的padding，类似same】  
-> 或者加个卷积把尺寸同意一下（可以后面尝试）

？？？？？？？？？？
输出全是一个色是什么鬼啊。。。  
明天先改回去，把FeatureExtraction后面还是直接返回输出结果。  
把前面分叉的model打印看一下

**20190405**  
remind:  
每次改完网络结构，有两个地方要改：  
parm.py中param.output_size，根目录utils.py中pad_width

- [ ] 查问题!!! 是数据生成的问题??【为啥loss降到0.03，mse和bad pixel起飞？？】 mvs_generator有问题么?
- [ ] cost volume 还没写。。

★ Keras中有Dense函数，可以直接用？







----------------------------
原始文档
```
1TODO:
    param.py
        --batch-size=16
        --input_size=23+2
        --output_size=input_size-22
        --idx_0d,idx_90d.....
        --model_name
        --model_conv_depth
        --model_filter_nums
        --model_learning_rate
        --logfile_dir = '/home/dell/Users/Samuel/Projects/epinet/epinet_checkpoints/' + model_name + 'train.txt'
        --is_train = True
        --checkpoint_filename = ''
        --steps_per_epoch = 10000
	data_process.py
	    --read_data(dir_list, idxs)
	        return raw_data_0d(n_scence,9,512,512,3), raw_data_90d.... , raw_label(n_scence,512,512)
	    --stack_imgs(raw_data)
	        return stacked_data_0d(n_scence, 512, 512, 9), stack_data_90d....
        --generate_train(raw_data...raw_label)
            "random gray, scale,"
            return traindata_batch_90d(batch_size,input_size,input_size,9)... train_label(batch_size,input_size,input_size)
        --aug_train(train_batch_...,train_label)
            "rotation flipping gamma"
            return train_batch_..., train_label
    train.py
        --epi_generator()
            yield train_batch..., train_label
```
