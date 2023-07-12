import requests as r
from bs4 import BeautifulSoup
import json
import uuid
import os
import multiprocessing as mp

class GenericScrapper:
    def __init__(self) -> None:
        pass

    def get_from_soup(self, soup, target = None, target_type = 'class_', find_all = True):
        if find_all: return soup.find_all(**{target_type: target})
        return soup.find(**{target_type: target})
        
    
    def get_soup(self, url):
        return BeautifulSoup(r.get(url).text, features="html.parser")

class DDDScrapper(GenericScrapper):
    initial_urls = {
        'tfgs': 'https://ddd.uab.cat/search?cc=tfg&ln=ca&jrec=1',
        'tesis': 'https://ddd.uab.cat/search?cc=tesis&ln=ca&jrec=1',
        'treballs': 'https://ddd.uab.cat/collection/trerecpro?ln=ca',
    }
    def __init__(self, initial_link: str = 'tfgs', output_folder: str = './tmp/ddd/') -> None:

        self.initial = self.initial_urls[initial_link]
        self.out = output_folder

        os.makedirs(output_folder, exist_ok = True)

    def get_documents_in_soup(self, soup):
        return self.get_from_soup(soup, target = 'moreinfo', target_type='class_')

    def get_data_in_document_from_url(self, url):

        document_soup = self.get_soup(url)
        content =  self.get_from_soup(document_soup, 'registre', target_type='class_', find_all=False)
        title = self.get_from_soup(document_soup, 'titol', find_all=False)

        return content, title

    def get_next_page(self, soup):
        next = self.get_from_soup(soup, 'següent', target_type='alt', find_all=False)
        if next is not None: next = next.parent['href']
        return next
    
    def process_document(self, href):
        content, title = self.get_data_in_document_from_url(href)
        formatted = f"{title}\n{content}"
        with open(os.path.join(self.out, f"{str(uuid.uuid4())}.html"), 'w') as handler: handler.write(formatted)

    def process_page(self, url, mp_thr = 0):

        sopa = self.get_soup(url)
        next = self.get_next_page(sopa)
        documents = [d.find('a')['href'] for d in self.get_documents_in_soup(sopa) if d.find('a') is not None]
        if mp_thr: raise NotImplementedError
        else: 
            for document in documents:
                self.process_document(document)

        return next


    def crawl(self):

        url = self.initial
        while url:
            url = self.process_page(url)

class PartituresScrapper(GenericScrapper):
    initial_urls = 'http://bdh.bne.es/bnesearch/Search.do?fechaFdesde=&showYearItems=&advanced=&exact=&textH=&completeText=&text=&sort=&languageView=es&fechaFhasta=&tipomaterial1=M%c3%basica+impresa&tipomaterial1=M%c3%basica+manuscrita&pageNumber=1&pageSize=30&language='
    def __init__(self, output_folder: str = './tmp/ddd/') -> None:
        self.out = output_folder

        os.makedirs(output_folder, exist_ok = True)

    def get_documents_in_soup(self, soup):
        return self.get_from_soup(soup, target = 'entrada', target_type='class_')
        

    def get_data_in_document_from_url(self, url):
        soup = self.get_soup(url)
        content =  self.get_from_soup(soup, target = 'entrada')
        pdfurl = self.get_from_soup(soup, target = 'identificador')[0]['href']
        print(pdfurl)

        return content, pdfurl

    def get_next_page(self, soup):
        next = self.get_from_soup(soup, 'sig', target_type='class_', find_all=False)
        if next is not None: next = 'http://bdh.bne.es' + next['href']
        return next
    
    def download_pages(self, url):
        soup = self.get_soup(url)
        return self.get_from_soup(soup, target = 'viewer-page-iframe')['src']

    def process_document(self, href):
        content, pdfurl = self.get_data_in_document_from_url(href)
        uid=str(uuid.uuid4())
        formated = f"{content}\n <a href='{pdfurl}'> pdf_url </a>"
        with open(os.path.join(self.out, f"{uid}.html"), 'w') as handler: handler.write(formated)
        

    def process_page(self, url, mp_thr = 0):

        sopa = self.get_soup(url)
        next = self.get_next_page(sopa)
        documents = ['http://bdh.bne.es' + self.get_from_soup(d, target='LabelBlueBold', target_type='class_')[0]['href'] for d in self.get_documents_in_soup(sopa)]
        if mp_thr: raise NotImplementedError
        else: 
            for document in documents:
                print(document)
                self.process_document(document)

        return next


    def crawl(self):

        url = self.initial_urls
        while url:
            url = self.process_page(url)
PartituresScrapper().crawl()