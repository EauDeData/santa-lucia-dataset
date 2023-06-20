from bs4 import BeautifulSoup
import requests
import uuid
import re
import os

BASE_URL = "https://enmemoria.lavanguardia.com"
INITIAL_URL = f"{BASE_URL}/esquelas?_fstatus=browse;date_limit=0;hdate=2023-06-18;type=all_memorial"
DATE_REGEX = r"\d{4}-\d{2}-\d{2}"
BASE_DB = "data"

def scrap_day(url):

    day = re.search(DATE_REGEX, url).group(0)
    html_folder = f"{BASE_DB}/{day}/htmls"
    imfolders = f"{BASE_DB}/{day}/images"

    os.makedirs(html_folder, exist_ok=True), os.makedirs(imfolders, exist_ok=True)
    
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, features="html.parser")

    esqueles = soup.find_all(class_ = "notice")
    for esquela in esqueles:

        try:
            page = esquela.find('img')
            if page is None: continue
            download_img = page['src'].replace('_medium', '_large')
            html_base = str(esquela)
            response = requests.get(download_img)

            file_id = uuid.uuid4()
            with open(f"{imfolders}/{file_id}.jpg", 'wb') as f: f.write(response.content)
            with open(f"{html_folder}/{file_id}.html", 'w') as f: f.write(html_base)
        except Exception as e: print(e)
        
    prev = soup.find(class_ = "prev").find("a")['href']
    return BASE_URL + prev, day

currently = INITIAL_URL
while True: 
    # TODO: Possar condició de parada com déu mana
    currently, day = scrap_day(currently)
    print(f"\tPeople dying in: {day}...\t", end = '\r')
