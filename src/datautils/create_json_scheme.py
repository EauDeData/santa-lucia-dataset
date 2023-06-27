import os
import json
from tqdm import tqdm
import layoutparser as lp
import tesserocr
from PIL import Image
import cv2
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pdfminer
from pdfminer.high_level import extract_pages
import matplotlib.pyplot as plt
from pypdf import PdfReader
import re
from multiprocessing import Manager
import numpy as np
import torch
import multiprocessing as mp
import easyocr

from src.process.segmentation import lp_detect, MODELS
from src.datautils.dataloader import read_img

BS = 10

def load_vocab():
    with open('./src/es.txt') as file:
        lines = file.readlines()
    return [line.strip().lower() for line in lines]
VOCAB = load_vocab()

def preprocess(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (5, 5), 1)
    _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image 

def ocr_img(img):
    return tesserocr.image_to_text(Image.fromarray(img), lang = 'spa')

def post_process(text):
    spplited = re.sub(r'[^\w\s]|_', ' ', text).split()
    newtext = []
    skip = False
    for n, word in enumerate(spplited[:-1]):
        if skip:
            skip = False
            continue

        comb = word + spplited[n+1]
        if  (comb.lower() in VOCAB) and (len(word) >= 2) and len(comb) > 4:
            
            newtext.append(comb)
            skip = True
        else: 
            newtext.append(word)
    return " ".join(newtext)

def extract_text_with_position(page_layout, page, max_x, max_y, x, y, x2, y2, returned = None, idx = None):
    text = ""

    for element in list(extract_pages(page_layout))[page]:
        if isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
            for text_line in element:
                x_pixels, y_pixels, _, _ = text_line.bbox
                arr_pixel_x = (x_pixels /  max_x) 
                arr_pixel_y = 1- (y_pixels / max_y)

                if (arr_pixel_x > x2 or arr_pixel_y > y2): continue
                for character in text_line:
                    if isinstance(character, pdfminer.layout.LTChar):
                        x_pixels, y_pixels, _, _ = character.bbox
                        
                        arr_pixel_x = (x_pixels /  max_x) 
                        arr_pixel_y = 1- (y_pixels / max_y)

                        if x <= (arr_pixel_x) <= x2 and y <= (arr_pixel_y) <= y2:
                            char = character.get_text()
                            text += char
    t = post_process(text)
    if returned is not None:
        returned[idx] = t
    
    return t

def save_file(fname):
    try:
        pre, _ = os.path.splitext(fname)
        pre = pre.replace('images', 'numpy')
        if os.path.exists(pre + '.npz'): return True
        files = read_img(fname, how = str)
        np.savez_compressed(pre + '.npz', **files)
    except: return True
    return True

def mp_manager_saver(files, t_number, n_threads):
    for img_idx in tqdm(range(t_number, len(files), n_threads)):
        save_file(files[img_idx])

def just_save_numpy(folder, mp_general = 6):
    file_extensions = ['.pdf',]
    print(f"Function triggered with origin {folder}")
    allfiles = []
    out = os.path.join(folder, 'numpy/')
    os.makedirs(out, exist_ok=True)

    for root, _, files in os.walk(folder):
        for file in files:
            if not (os.path.splitext(file)[1].lower() in file_extensions): continue

            fname = os.path.join(root, file)
            allfiles.append(fname)
            
            print(f"Appending file number... {len(allfiles)}\t\t", end = '\r')
    print('\nAll files added, launching task!\n')

    common = Manager()
    list_files = common.list(allfiles)
    process = [mp.Process(target=mp_manager_saver, args=(list_files, t, mp_general)) for t in range(mp_general)]
    [p.start() for p in process]
    [p.join() for p in process]


            

def paralel_extract_wrapper(args):
    return extract_text_with_position(*args)

def mp_extract(list_of_args, thread_num, num_threads, returned):
    for idx in range(thread_num, len(list_of_args), num_threads): extract_text_with_position(*list_of_args[idx], returned, idx)
    return True


def process_folder(folder, out_base, LPMODEL, mp_ocr = 0, ocr = True, ocr_device = 'cuda', margin = 10, file_extensions = ['.pdf',]):
    
    print(f"Function triggered with origin {folder} and destination {out_base}")

    out = out_base
    os.makedirs(out, exist_ok=True)
    if ocr: reader = easyocr.Reader(['es'], gpu = ocr_device)

    for root, _, files in os.walk(folder):
        for file in tqdm(files, desc=f"Processing {folder}..."):
            outname = os.path.join(out, file.lower().replace('.pdf', '.json'))

            if not os.path.splitext(file)[1].lower() in file_extensions: continue
            if os.path.exists(outname): continue

            fname = os.path.join(root, file)
            try:
                images = np.load(fname.replace('images', 'numpy').replace('.pdf', '.npz'))
            except FileNotFoundError: images = read_img(fname)
            images = [images[x] for x in images]
            json_gt = {
                    "file": file, 
                    "path": fname,
                    "pages": {}
                    }
            pdfhandler = PdfReader(fname).pages

            for num, image in enumerate(images):

                json_gt["pages"][num] = []
                with torch.no_grad():
                    detection = lp_detect(image, LPMODEL)

                returned = [None] * len(detection)
                if mp_ocr: 
                    m = Manager()
                    returned = m.list(returned)

                _, _, max_x, max_y = pdfhandler[num].mediabox
                
                if mp_ocr:
                    crops = []
                
                if ocr:
                    image = preprocess(image)

                for mp_num, item in enumerate(detection):
                    

                    json_gt["pages"][num].append(
                        {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                    )
                    x,y,w,h = [int(x) for x in item.coordinates]
                    crop = image[y:max(h-1,0), x:max(w-1, 0)]
                    if ocr: 
                        text = reader.readtext(crop)
                        print(text)
                        returned[mp_num] = text
                        continue

                    else:
                        x, y = x-margin, y - margin  
                        w,h = w+margin, h+margin
                        if not mp_ocr:
                            # text =   tesserocr.image_to_text(crop, lang = 'spa')  #OCR.readtext(crop)
                            text = extract_text_with_position(fname, num, max_x, max_y, x = x / image.shape[1], y= y/image.shape[0], x2=w/image.shape[1], y2=h/image.shape[0])
                            
                            returned[mp_num] = text
                        else: crops.append((fname, num, max_x, max_y, x / image.shape[1], y/image.shape[0], w/image.shape[1], h/image.shape[0])) 
                    
                    if mp_ocr and not ocr:
                        process = [mp.Process(target = mp_extract, args=(crops, i, mp_ocr, returned)) for i in range(mp_ocr)] # TODO: fix it this aint doing shit
                        [p.start() for p in process]
                        [p.join() for p in process]

                for mp_num, element in enumerate(returned):
                    if element is not None: json_gt["pages"][num][mp_num]['ocr'] = element
            
            json.dump(dict(json_gt), open(outname, 'w')) # TODO: Ensure it arrives here on join            
    del LPMODEL


def rescale(image, x, y, w, h, model):
    longest = max(image.shape)
    if longest < model.cfg.INPUT.MAX_SIZE_TEST: return x, y, w, h
    scale = longest / model.cfg.INPUT.MAX_SIZE_TEST 
    return int(scale * x), int(scale * y), int(scale * w), int(scale * h)

    
