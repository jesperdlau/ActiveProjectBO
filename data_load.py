
import os
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np 

folder_path = os.getcwd() + "\data\shipsnet\shipsnet"

img_list = []
label_list = []

for filename in os.listdir(folder_path):
    f = os.path.join(folder_path,filename)
    if os.path.isfile(f):
        img_list.append(imread(f))
        label_list.append(int(filename[0]))


np.save("image_data.npy",np.array(img_list))
np.save("labels.npy",np.array(label_list))

# print(np.shape(img_list[0]))
# plt.imshow(img_list[0])
# plt.show()

