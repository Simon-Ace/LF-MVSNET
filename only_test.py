import matplotlib.pyplot as plt
import numpy as np

import imageio
import os

LF_train_dir = '/home/dell/User/shuo/dataset/HCI4D/additional/'

raw_data_90d = np.zeros(shape=(512, 512, 3, 9, 2), dtype=np.float32)
raw_data_90d[:, :, :, 0, 0] = np.float32(imageio.imread(os.path.join(LF_train_dir,'antinous/') + '/input_Cam0%02d.png' % 40))

plt.figure()
vis2 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
vis2[:,:,:] = raw_data_90d[:, :, :, 0, 0]
imageio.imwrite(r'1.jpg',vis2)

# import numpy as np
# from matplotlib import pyplot as plt
#
# from skimage import data
#
# random_image = np.random.random([500, 500])
# print(random_image)
# plt.imshow(random_image, cmap='gray')
# plt.colorbar()
# plt.show()
