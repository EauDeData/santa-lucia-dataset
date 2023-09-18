import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

webdriver.firefox.marionette=False
# El rows = 1000 travieso és per aprofitar el bug
BASE_ARXIU = 'https://arxiusenlinia.cultura.gencat.cat'
QUERY_GLOBAL = BASE_ARXIU +'/#/cercaavancada/llistatCercaAvanc'

option = webdriver.ChromeOptions()
option.add_experimental_option("excludeSwitches", ["enable-automation"])
option.add_experimental_option('useAutomationExtension', False)

#For ChromeDriver version 79.0.3945.16 or over
option.add_argument('--disable-blink-features=AutomationControlled')

option.add_argument("window-size=1280,800")
option.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")

DRIVERPATH = 'geckodriver' 
OUPATH = 'arxiu/'
DRIVER = webdriver.Chrome(options=option)
DRIVER.get(QUERY_GLOBAL)

DRIVER.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

input("Press enter to proceed... ") # Cal inserir la cerca manualment perque funciona desde front-end; tirem de selenium
CSV = open('imatges.tsv', 'w')
CSV.write('\t'.join(['url', 'caption', 'desc']))
visited = set()

def check_if_loading():
    try:
        return DRIVER.find_element(By.CLASS_NAME, 'spinner-container').is_displayed
    except NoSuchElementException:
        return False

while True:
    # while check_if_loading(): pass

    for element in BeautifulSoup(DRIVER.page_source, features="lxml").find('table').find_all(class_ = 'row'):
        soup_elelment =element # METADATA FROM HERE
        photo = soup_elelment.find('img', class_ = 'image-list-item-responsive')
        if photo is not None:
            if photo['src'] in visited: continue # Cal fer-ho en dos nivells perque si és None no és iterable per tant no té "in" 
        desc = soup_elelment.find_all(class_ = 'contingut-nom-metadada')
        if not len(desc): continue
        desc = desc[-1].find_next_siblings('span')
        title = soup_elelment.find(class_ = 'titol-text-resultats')
        if photo is not None and title is not None and desc is not None:
            photo = photo['src']
            title = title
            desc = desc[0]

            data = {
                'image': str(photo),
                'caption': str(title).replace('\n', ' '),
                'desc': str(desc).replace('\n', ' ')
            }
            visited.add(photo)
            
            CSV.write("\n" + '\t'.join(list(data.values()))) # Separat per tabulacions per evitar problemes de parsing amb les commes de les fotos i descripcions
    succeed = False
    while not succeed:
        # De vegades el botó de següent está oclòs ja que BS4 va més ràpid que selenium carregant les coses, per algun motiu comprobar el display no funciona
        try:
            next = WebDriverWait(DRIVER, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'ui-paginator-next'))
            )
            next.click()
            succeed = True
        except: pass


            