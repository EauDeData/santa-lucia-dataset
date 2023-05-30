import os
import json
from tqdm import tqdm
import layoutparser as lp
import pytesseract
from pytesseract import Output

from src.process.segmentation import lp_detect, MODELS
from src.datautils.dataloader import read_img

BS = 10

def process_folder(folder, out_base, ocr = True):
    file_extensions = ['.pdf',]
    print(f"Function triggered with origin {folder} and destination {out_base}")

    LPMODEL = lp.models.Detectron2LayoutModel(
            config_path =MODELS["prima"]['model'], # In model catalog
            label_map   = MODELS["prima"]["labels"], # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
        )

    out = out_base
    os.makedirs(out, exist_ok=True)
    for root, _, files in os.walk(folder):
        for file in tqdm(files, desc=f"Processing {folder}..."):
            if not os.path.splitext(file)[1].lower() in file_extensions: continue
            
            fname = os.path.join(root, file)
            outname = os.path.join(out, file.lower().replace('.pdf', '.json'))
            images = read_img(fname)
            json_gt = {
                    "file": file, 
                    "path": fname,
                    "pages": {}
                    }
            
            for num, image in enumerate(images):

                json_gt["pages"][num] = []
                for item in lp_detect(image, LPMODEL):

                    json_gt["pages"][num].append(
                        {"type": item.type, "bbox": [int(x) for x in item.coordinates], 'conf': item.score}
                    )
                    if ocr:
                        
                        x,y,w,h = json_gt["pages"][num][-1]["bbx"]
                        json_gt["pages"][num][-1]["tesseract_ocr"] = pytesseract.image_to_data(image[y:y+h, x:x+w], lang = 'spa', nice = True, output_type=Output.DICT)

            json.dump(json_gt, open(outname, 'w'))
    del LPMODEL
