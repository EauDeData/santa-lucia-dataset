import os
import json
from tqdm import tqdm
import layoutparser as lp
import pytesseract
import tesserocr
from pytesseract import Output
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
from detectron2.data.transforms import ResizeShortestEdge
import numpy as np
import multiprocessing as mp

from src.process.segmentation import lp_detect, MODELS
from src.datautils.dataloader import read_img

BS = 10
def process_folder(folder, out_base, LPMODEL, ocr = True,):
    file_extensions = ['.pdf',]
    print(f"Function triggered with origin {folder} and destination {out_base}")
    TR = ResizeShortestEdge([LPMODEL.cfg.INPUT.MIN_SIZE_TEST, LPMODEL.cfg.INPUT.MIN_SIZE_TEST], LPMODEL.cfg.INPUT.MAX_SIZE_TEST)


    out = out_base
    os.makedirs(out, exist_ok=True)
    for root, _, files in os.walk(folder):
        for file in tqdm(files, desc=f"Processing {folder}..."):
            outname = os.path.join(out, file.lower().replace('.pdf', '.json'))

            if not os.path.splitext(file)[1].lower() in file_extensions: continue
            if os.path.exists(outname): continue

            fname = os.path.join(root, file)
            images = read_img(fname)
            manager = mp.Manager()
            json_gt = {
                    "file": file, 
                    "path": fname,
                    "pages": {}
                    }
            for num, image in enumerate(images):

                json_gt["pages"][num] = []
                detection = lp_detect(image, LPMODEL)

                parameters_ocr = []
                returned = manager.list([None] * len(detection))

                for mp_num, item in enumerate(detection):
                    

                    json_gt["pages"][num].append(
                        {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                    )
                    if ocr:
                        
                        parameters_ocr.append((json_gt["pages"][num][-1]["bbox"], image, returned, mp_num))
                
            with mp.Pool(8) as p: p.starmap(_ocr_paralel, parameters_ocr)
            if ocr:
                for mp_num, item in enumerate(detection):
                    element = returned[mp_num]
                    if element is not None: json_gt["pages"][num][mp_num]['ocr'] = element
            json.dump(dict(json_gt), open(outname, 'w')) # TODO: Ensure it arrives here on join
    del LPMODEL

def _ocr_paralel(box, image, list_mp, num):
    x,y,w,h = box
    try:
        input_image = Image.fromarray(image[y:max(h-1,0), x:max(w-1, 0)])
        list_mp[num] = tesserocr.image_to_text(input_image, lang = 'spa') # pytesseract.image_to_data(image[y:y+h, x:x+w], lang = 'spa', nice = 10, output_type=Output.DICT)

    except ValueError:  pass # This is just a too small region, don't worry I'm an engineer

def rescale(image, x, y, w, h, model):
    longest = max(image.shape)
    if longest < model.cfg.INPUT.MAX_SIZE_TEST: return x, y, w, h
    scale = longest / model.cfg.INPUT.MAX_SIZE_TEST 
    return int(scale * x), int(scale * y), int(scale * w), int(scale * h)

    