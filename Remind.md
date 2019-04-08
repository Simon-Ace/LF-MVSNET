**做的修改：**
- [x] 图片数量 9 -> 5， 【Epoch 142 -> loss 0.0256】
- [ ] 图片大小 25 -> 64 【loss、mse降的快，图片细节不好，还会有纹理扩散】 out
- [x] 数量 5 -> 3，[1 3 5] out_new 【loss到0.1左右就很慢了】  ||or [1 2 3] out_123 【loss 0.1左右很慢】
- [ ] 两张图[1,5] out_two_pic  [gpu 4]



**20190408**

这个是算什么的？  
根目录`utils.py` `result[0:482,:] = np.uint8(25*np.reshape(np.transpose(train_label482,(1,0,2)),(482,sz*482))+100)`