import cv2
import numpy as np
import argparse
from visualize import *

def gray_world(img):
    """
    IN: input img => shape: (256, 256, 3)
    OUT: white balanced img => shape: (256, 256, 3)
    """
    
    # Write your code here!
    img = img.transpose(2, 0, 1).astype(np.uint32) # (1024, 1024, 3) => (3, 1024, 1024)
    print(np.max(img[1]))
    R_AVG = np.mean(img[0])
    G_AVG = np.mean(img[1])
    B_AVG = np.mean(img[2])
    img[0] = np.minimum((G_AVG/R_AVG)*img[0],1023)
    img[2] = np.minimum((G_AVG/B_AVG)*img[2],1023)
    img = np.clip(img, 0, 1023)
    return img.transpose(1, 2, 0).astype(np.uint16) # (3, 1024, 1024) => (1024, 1024, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_root', type=str, default='data/')
    parser.add_argument('--sample_num', type=str, default='1')
    args = parser.parse_args()

    file_name = f"sample_{args.sample_num}"

    print()
    print("Read Images..", end=' ')
    img_input = cv2.imread(os.path.join(args.sample_root, file_name+'.tiff'), cv2.IMREAD_UNCHANGED)
    img_gt = cv2.imread(os.path.join(args.sample_root, file_name+'_gt.tiff'), cv2.IMREAD_UNCHANGED)
    print("Done!")

    print("White Balance using Gray World Algorithm..", end=' ')
    img_wb = gray_world(img_input)
    print("Done!")

    print("Visualize Images..", end=' ')
    vis_result = visualize(img_input, img_wb, img_gt) 
    cv2.imwrite('results/gray_world.png',vis_result)
    print("Done!")
    print()