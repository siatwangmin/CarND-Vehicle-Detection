import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

img = mpimg.imread('test_images/test1.jpg')
plt.imshow(img)
plt.show()

# output_image, bboxes = apply_sliding_window(img, svc, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins)
# #     plt.imshow(output_image)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# f.subplots_adjust(hspace = .2, wspace=.05)
# ax1.imshow(img)
# ax1.set_title('Origin', fontsize=30)
# ax2.imshow(output_image)
# ax2.set_title('Result Images', fontsize=30)