import os
import cv2
import pdf2image
import numpy as np
import cv2
import zipfile
import warnings
def read_img(path):
    img = pdf2image.convert_from_path(path.strip())
    return {i: np.array(img[i]) for i in range(len(img))}

class ZippedDataloader:
    def __init__(self, path_to_zip, temporal_folder = './.tmp/') -> None:
        os.makedirs(temporal_folder, exist_ok=True)

        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(temporal_folder)
        
        file_extensions = ['.pdf',]

        self.files = []
        for root, _, files in os.walk(temporal_folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in file_extensions:
                    self.files.append(os.path.join(root, file))
        self.inner_state = 0

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return read_img(self.files[index])
    
    def __next__(self):
        
        if self.inner_state > (len(self) - 1):
            self.inner_state += 1
            return self[self.inner_state - 1]
        
        raise StopIteration