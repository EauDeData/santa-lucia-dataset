import torch
import clip
from PIL import Image
import numpy as np
import shutil
import os
import argparse
import tesserocr
import json
class CLIPClassifier:
    def __init__(self, labels = ['An image showing some text', 'Some rendered text document', 'Textual data', 'text', 'text with some logos', 'An dark image of white flowers', 'rendered photography', 'A person holding a lit candle in their hands, an scp anomalous object, black background pinterest'], device = 'cpu') -> None:
        self.labels = labels

        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.tokenized = clip.tokenize(labels).to(self.device)

    def forward(self, image: Image):
        with torch.no_grad():
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            logits_per_image, logits_per_text = self.model(image, self.tokenized)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return np.argmax(probs[0])

def process_ocr(pil_image):
    return tesserocr.image_to_text(pil_image, lang = 'spa')

def find_rejects(base_folder, clip_classifier, target = [0, 1, 2, 3, 4], extensions = ['.jpg']):
    file_counter = 0
    reject_counter = 0

    for m, (root, _, files) in enumerate(os.walk(base_folder)):
        for n, file in enumerate(files):
            print(f"Folder num {m} file num {n}...", end = '\r')
            if os.path.splitext(file)[1].lower() in extensions:
                fullpath = os.path.join(root, file)
                image = Image.open(fullpath)
                #label = clip_classifier.forward(image)
                w, h = image.size
                if (w == 178 and h == 178): # Double Check rejection idk
                    shutil.copy(fullpath, os.path.join('./rejects', file))
                    reject_counter += 1
                else:
                    text = {"file": file, "path": fullpath, "ocr": process_ocr(image)}
                    os.makedirs(root.replace('images', 'jsons'), exist_ok=True)
                    json.dump(text, open(fullpath.replace('images', 'jsons').replace('.jpg', '.json'), 'w'))


                file_counter += 1

    print(f'\nFinished with {file_counter} found, {reject_counter} rejected. Total: {file_counter - reject_counter}')
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='Lorem Ipsum',
                        description='Super Trouper',
                        epilog='Uwu')

    parser.add_argument('-w', '--where_data', default="./data")
    args = parser.parse_args()

    clippy = None
    find_rejects(args.where_data, clippy)