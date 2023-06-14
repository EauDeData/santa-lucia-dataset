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
            json_gt = {
                    "file": file, 
                    "path": fname,
                    "pages": {}
                    }
            
            for num, image in enumerate(images):

                json_gt["pages"][num] = []
                detection = lp_detect(image, LPMODEL)


                for item in detection:
                    

                    json_gt["pages"][num].append(
                        {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                    )
                    if ocr:
                        
                        x,y,w,h = json_gt["pages"][num][-1]["bbox"]
                        try:
                            input_image = Image.fromarray(image[y:max(h-1,0), x:max(w-1, 0)])
                            json_gt["pages"][num][-1]["ocr"] = tesserocr.image_to_text(input_image, lang = 'spa') # pytesseract.image_to_data(image[y:y+h, x:x+w], lang = 'spa', nice = 10, output_type=Output.DICT)
                        except ValueError:  continue # This is just a too small region, don't worry I'm an engineer
            json.dump(json_gt, open(outname, 'w'))
    del LPMODEL

def rescale(image, x, y, w, h, model):
    longest = max(image.shape)
    if longest < model.cfg.INPUT.MAX_SIZE_TEST: return x, y, w, h
    scale = longest / model.cfg.INPUT.MAX_SIZE_TEST 
    return int(scale * x), int(scale * y), int(scale * w), int(scale * h)

    