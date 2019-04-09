**做的修改：**
- [x] 图片数量 9 -> 5， 
    - 【Epoch 142 -> loss 0.0256, 25x25】
    - 【folder: out1-5pic-input-loss_0.025-效果还凑合】
- [x] 图片数量5，图片大小 25 -> 64 
    - 【loss、mse降的快，图片细节不好，还会有纹理扩散】 
    - 【mse-0.617_bp-7.20_trainLoss-0.0207，最后看效果还不错，无纹理地区有的不太好，会出现空洞？】
    - folder: out2-5pic-Size64x64-loss_0.02-还可以
- [x] 数量 5 -> 3，[1 3 5] out_new 【loss到0.1左右就很慢了】  ||or [1 2 3] out_123 【loss 0.1左右很慢】
- [x] 两张图[1,5] out_two_pic  [gpu 3] 【Epoch 110 -> loss 0.05，mse-4.884_bp-22.94，图片效果也不太好】



**20190408**

这个是算什么的？  
根目录`utils.py` `result[0:482,:] = np.uint8(25*np.reshape(np.transpose(train_label482,(1,0,2)),(482,sz*482))+100)`



- [x] 加了cost volume，在generate_train_data()训练 和 stack_data()测试中
    - 训练集效果好mse 0.087；bp 1.11，测试集不行，好多地方效果不好