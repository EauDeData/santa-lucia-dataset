from src.datautils.dataloader import ZippedDataloader
from src.process.segmentation import segment, crop_lines, lp_detect

import argparse
import uuid
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import layoutparser as lp
import numpy as np

parser = argparse.ArgumentParser(
                    prog='Lorem Ipsum',
                    description='Super Trouper',
                    epilog='Uwu')

parser.add_argument('-f', '--file', default='./diaris.zip')
parser.add_argument('-o', '--output', default='./cropped/')
args = parser.parse_args()
os.makedirs(args.output, exist_ok = True)

dataset = ZippedDataloader(args.file)

for n, image in tqdm(enumerate(dataset,)):
    #image = image.transpose(1, 0, 2)
    layout = lp_detect(image[0])
    for item in layout:
        print(item.block)
        print(item.type, item.coordinates, item.score)
    vis = np.array(lp.visualization.draw_box(image[0], layout)) #.transpose(1, 0, 2)
    plt.imshow(vis)
    plt.show()

