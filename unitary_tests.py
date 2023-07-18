from src.datautils.dataloader import ZippedDataloader, read_img
from src.process.segmentation import segment, crop_lines, lp_detect, MODELS

import argparse
import uuid
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import layoutparser as lp
import numpy as np


image = np.load('/data2fast/users/amolina/BOE/republica_ii/numpy/00a3f6a1-02a1-4358-9f7e-8d022b33c5f5.npz')['0']
for model_name in MODELS:
    model = lp.models.Detectron2LayoutModel(config_path = MODELS[model_name]['model'], label_map   = MODELS[model_name]['labels'], extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
    img = lp.visualization.draw_box(image, lp_detect(image, model), color_map = {})
    cv2.imwrite(f'results/{model_name}-lp.png', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
print('im here')
model = lp.models.Detectron2LayoutModel(config_path = MODELS['hjdataset']['model'], label_map   = MODELS['hjdataset']['labels'], extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
img = lp.visualization.draw_box(image.transpose(1, 0, 2), lp_detect(image.transpose(1, 0, 2), model), color_map = {})
cv2.imwrite(f'results/hjdataset-lp.t.png', cv2.cvtColor(np.array(img).transpose(1, 0, 2), cv2.COLOR_RGB2BGR))

image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bbxs = segment(image_gs)

canvas = image.copy()
for box in bbxs:
    if not isinstance(box, int): canvas = cv2.rectangle(canvas, box[:2], [a+w for a,w, in zip(box[2:], box[:2])], (255, 0, 0), 5)

cv2.imwrite(f'results/ours.png', cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
