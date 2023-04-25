from src.datautils.dataloader import ZippedDataloader
from src.process.segmentation import segment, crop_lines

import argparse
import uuid
from tqdm import tqdm
import os
import cv2
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
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    subfolder = f"{args.output}{n}/"
    os.makedirs(subfolder, exist_ok = True)
    bbxs = segment(image)
    
    for bbx in bbxs:
        
        if not isinstance(bbx, int):
            fname = f"{subfolder}{uuid.uuid4()}.png"
            crop_lines(image[bbx[1]:bbx[1] + bbx[3], bbx[0]:bbx[0] + bbx[2]], fname)
