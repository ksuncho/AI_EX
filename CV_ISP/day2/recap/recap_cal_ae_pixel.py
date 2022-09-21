import cv2
import numpy as np

img1 = cv2.imread('sample_1.png') # (256, 256, 3) value 5~254
img2 = cv2.imread('sample_2.png') # (256, 256, 3) value 5~254

# Write your code for calculating pixel-level mae

# normalize every pixel to be unit vector
img1_normalized_per_pixel = img1 / np.linalg.norm(img1, axis=2)[:, :, None]
img2_normalized_per_pixel = img2 / np.linalg.norm(img2, axis=2)[:, :, None]

# result of dot product of unit vectors == cos_sim_map
cos_sim_map = np.sum(img1_normalized_per_pixel * img2_normalized_per_pixel, axis=2)

# convert cos similarity to angular error 
ae_map = np.arccos(cos_sim_map) * 180 / np.pi

mae = np.mean(ae_map)

print(mae)