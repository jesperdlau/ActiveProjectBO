import os
from skimage import io, color, morphology
import matplotlib.pyplot as plt
import numpy as np 
import cv2

# Linux
# folder_path = "ActiveProjectBO/data/shipsnet/shipsnet"

# Windows
folder_path = os.getcwd() + "\data\shipsnet\shipsnet"

img_list = []
label_list = []
img_list_processed = []

def process(img):
    # Border median
    img_border = np.concatenate((img[:,0], img[:,-1], img[0,:], img[-1,:]))
    bm = np.median(img_border, axis=0)
    
    # Color thresholding
    dif = 15
    thr = cv2.inRange(img[:,:,0], bm[0]-dif, bm[0]+dif)
    thg = cv2.inRange(img[:,:,1], bm[1]-dif, bm[1]+dif)
    thb = cv2.inRange(img[:,:,2], bm[2]-dif, bm[2]+dif)
    thcomb = thr & thg & thb
    mask = np.logical_not(thcomb)
    
    # Remove small objects
    mask_nonoise = morphology.remove_small_objects(mask, 10)
    img_th = cv2.bitwise_and(img,img, mask= np.uint8(mask_nonoise))

    # RGB to Gray
    img_gray = color.rgb2gray(img_th)

    return img_gray


# Main loop
for i, filename in enumerate(os.listdir(folder_path)):
    file = os.path.join(folder_path,filename)
    if os.path.isfile(file):
        img = io.imread(file)
        img_list.append(img)
        label_list.append(int(filename[0]))
        img_list_processed.append(process(img))
        print(i)

# io.imshow_collection(img_list_processed[21:30], cmap="gray")
# plt.show()
# print()

np.save("image_data.npy",np.array(img_list))
np.save("labels.npy",np.array(label_list))
np.save("image_data_preprocessed.npy", np.array(img_list_processed))

