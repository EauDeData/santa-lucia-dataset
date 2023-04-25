import cv2
import numpy as np
import math

import multiprocessing as mp



def ad_contours_mp(boxes_bbx, contours, hierarchy, thread, num_threads):
        
    for n in range(thread, len(contours), num_threads):
        c = contours[n]
        if hierarchy[0][n][-2] == -1: continue
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        x,y,w,h = rect

        boxes_bbx[n] = (x, y, w, h)

    return None

def segment(img):
    # Processing operations
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) # Letters (1) Background (0)
    dst = cv2.cornerHarris(thresh,10,3,0.04)
    dst = (dst - dst.min()) / (dst.max() - dst.min()) * 255
    ret, thresh = cv2.threshold(dst.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    blops = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, blops)
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, horizontalStructure.T)

    contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes_bbx = [0 for _ in range(len(contours))]
    threads = 8

    L = mp.Manager().list(boxes_bbx)
    processes = [mp.Process(target = ad_contours_mp, args=(L, contours, hierarchy, i, threads)) for i in range(threads)]
    [p.start() for p in processes]
    [p.join() for p in processes]


    return list(L)

def crop_lines(img, imname = None):
    
    dst = cv2.Canny(~img, 50, 200, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 200, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho

            if round(a/b, 7) == 0 and y0 < dst.shape[0] * .75 and y0 > dst.shape[0] * .25:
                y0 = int(y0)
                im1 = img[:y0, :]
                im2 = img[y0:, :]

                base = imname.split('.')[0]
                cv2.imwrite(f"{base}-1.png", im1)
                cv2.imwrite(f"{base}-2.png", im2)
                return True
    cv2.imwrite(imname, img)