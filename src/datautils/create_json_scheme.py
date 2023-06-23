import os
import json
from tqdm import tqdm
import layoutparser as lp
import tesserocr
from PIL import Image
import cv2
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from src.process.segmentation import lp_detect, MODELS
from src.datautils.dataloader import read_img

BS = 10



def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (5, 5), 1)
    return image

def ocr_img(img):
    return tesserocr.image_to_text(Image.fromarray(img), lang = 'spa')

def process_folder(folder, out_base, LPMODEL, mp_ocr = 0):
    file_extensions = ['.pdf',]
    print(f"Function triggered with origin {folder} and destination {out_base}")

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

                returned = manager.list([None] * len(detection)) if mp_ocr else [None] * len(detection)
                image = preprocess(image)
                if mp_ocr: crops = []

                for mp_num, item in enumerate(detection):
                    

                    json_gt["pages"][num].append(
                        {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                    )
                    x,y,w,h = [int(x) for x in item.coordinates]
                    crop = image[y:max(h-1,0), x:max(w-1, 0)]

                    if not mp_ocr:
                        text =   tesserocr.image_to_text(crop, lang = 'spa')  #OCR.readtext(crop)
                        returned[num] = text
                    else: crops.append(crop)
                
                else:
                    with ProcessPoolExecutor(max_workers=mp_ocr) as executor:
                        tasks = {executor.submit(ocr_img, img): n for n, img in enumerate(crops)}
                        for future in concurrent.futures.as_completed(tasks):
                            crop_number = tasks[future]
                            returned[crop_number] = future.result()

                for mp_num, element in enumerate(returned):
                    if element is not None: json_gt["pages"][num][mp_num]['ocr'] = element
            
            json.dump(dict(json_gt), open(outname, 'w')) # TODO: Ensure it arrives here on join            
    del LPMODEL


def rescale(image, x, y, w, h, model):
    longest = max(image.shape)
    if longest < model.cfg.INPUT.MAX_SIZE_TEST: return x, y, w, h
    scale = longest / model.cfg.INPUT.MAX_SIZE_TEST 
    return int(scale * x), int(scale * y), int(scale * w), int(scale * h)

    