from bs4 import BeautifulSoup
import requests
import uuid
import re
import os
import multiprocessing as mp

import json
from tqdm import tqdm

LINKS = json.load(open('gazeta.json', 'r'))
BASELINK = 'https://www.boe.es'
BASEQUERY = 'https://www.boe.es/buscar/'
BASE = './BOE'

def scrap_hard(url, folder):
    
    html_folder = f"{folder}/htmls"
    imfolders = f"{folder}/images"
    os.makedirs(html_folder, exist_ok=True), os.makedirs(imfolders, exist_ok=True)

    next_ = 1
    newurl = url
    while not next_ is None:

        html_text = requests.get(newurl).text
        soup = BeautifulSoup(html_text, features="html.parser")
        next_ = soup.find(class_ = "pagSig")

        items = soup.find_all(class_ = "resultado-busqueda")

        for item in tqdm(items):

            file_id = uuid.uuid4()            
            pdfurl = item.find(class_ = "puntoPDF2")
            if pdfurl is None: continue

            pdfurl = pdfurl.find("a")
            response = requests.get(BASELINK +pdfurl['href'])

            with open(f"{imfolders}/{file_id}.pdf", 'wb') as f: f.write(response.content)
            with open(f"{html_folder}/{file_id}.html", 'w') as f: f.write(str(item))

        
        ### NEXT PAGE ###
        if not next_ is None: newurl = BASEQUERY + next_.parent['href']


def scrap_easy(url, folder):
    html_folder = f"{folder}/htmls"
    imfolders = f"{folder}/images"
    os.makedirs(html_folder, exist_ok=True), os.makedirs(imfolders, exist_ok=True)

    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, features="html.parser")

    items = soup.find_all(class_ = "resultado-busqueda")

    for item in tqdm(items):

        file_id = uuid.uuid4()            
        pdfurl = item.find(class_ = "puntoPDF2")
        if pdfurl is None: continue

        pdfurl = pdfurl.find("a")
        response = requests.get(BASELINK +pdfurl['href'])

        with open(f"{imfolders}/{file_id}.pdf", 'wb') as f: f.write(response.content)
        with open(f"{html_folder}/{file_id}.html", 'w') as f: f.write(str(item))

def scrap_period(period):

    url, needs_next = LINKS[period]['link'], LINKS[period]['needs_next']
    out_folder = f"{BASE}/{period}"
    os.makedirs(out_folder, exist_ok= True)

    if needs_next: scrap_hard(url, out_folder)
    else: scrap_easy(url, out_folder)

def take_care(periods, thread_id, n_threads = 8):

    for idx in range(thread_id, len(periods), n_threads):
        print(f"I am thread {thread_id} and I'm taking care of {periods[idx]}")

        scrap_period(periods[idx])

periods = list(LINKS.keys())
n_jobs = 8
jobs = [mp.Process(target = take_care, args = (periods, i, n_jobs)) for i in range(n_jobs)]
[j.start() for j in jobs]
[u.join() for u in jobs]