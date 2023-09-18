
import json 
import argparse
import os
import layoutparser as lp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup
import re
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from src.process.nlp_utils import StringCleanAndTrim, sentence_proximity, SentenceTransformer, spanish, SequenceTagger, ner_detection

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='Lorem Ipsum',
                        description='Super Trouper',
                        epilog='Uwu')

    parser.add_argument('-wj', '--where_json', default="/home/amolina/Desktop/datasets/tmp/")
    parser.add_argument('-w', '--where_data', default="/data1tbsdd/BOE/")
    parser.add_argument('-t', '--threads', default=2, type=int)
    parser.add_argument('-o', '--overwrite', default=False, type=bool)
    parser.add_argument('-d', '--device', default="cuda")
    parser.add_argument('-k', '--top_k_similar', default=0.4, type=float)

    args = parser.parse_args()
    st = SentenceTransformer(spanish).to(args.device)
    tagger = SequenceTagger.load("flair/ner-spanish-large")
    max_sims = []
    for root, _, files in os.walk(args.where_json):
        if '_gt' in root: continue
        if 'jsons' in root and not 'gt' in root:
            os.makedirs(root.replace('jsons', 'jsons_gt_bis'), exist_ok=True)
        print(root, '\n')
        for n, file in enumerate(files):
            print('File number', n, '\t', end = '\r')
            if not (os.path.splitext(file)[1].lower() in ['.json']): continue
            filename = os.path.join(root, file)
            filename_out = filename if args.overwrite else filename.replace('.json', '_gt.json')
            if 'jsons' in root and not args.overwrite: filename_out = filename_out.replace('/jsons', '/jsons_gt_bis')
            if os.path.exists(filename_out): continue
            data = json.load(open(filename))

            if 'topic_gt' in data: continue # already done

            html_path = data['path'].replace('.pdf', '.html').replace('/data2fast/users/amolina/BOE/', args.where_data).replace('images', 'htmls')
            with open(html_path, 'r') as html_handler:

                ### Parsing Metadata ###
                text = ''.join(html_handler.readlines())
                soup = BeautifulSoup(text, features="lxml")
                query = '\n'.join(soup.find('h4').contents)

                a_tag = soup.select_one('li.puntoPDF2 > a')

                # Extract the href value
                href = a_tag['href']
                dd_tag = soup.find('dd')
                line = dd_tag.text.strip()
                date_pattern = r'\d{2}/\d{2}/\d{4}'
                match = re.search(date_pattern, line)
                match = None if match is None else match.group(0)

                data['date'] = match
                data['query'] = query
                data['document_href'] = href
            
            sentences = []
            route_to_gt = []
            sims = []
            for page in data['pages']:
                for num, item in enumerate(data['pages'][page]):
                    if not 'ocr' in item: continue
                    if len(item['ocr']) >= len(query):
                        sentences.append(item['ocr'])
                        route_to_gt.append({
                            'page': page,
                            'idx_segment': num
                        })
                        similarity = sentence_proximity(query, [item['ocr']], st)
                        sims.extend(similarity)
                        
                        data['pages'][page][num]['similarity'] = similarity[0]
            if not len(sentences): continue
           
            max_sims.append(max(sims))
            # if max(sims) >= args.top_k_similar:
            data['topic_gt'] = route_to_gt[np.argmax(sims)]
            data['ocr_gt'] = sentences[np.argmax(sims)]
            data['score'] = max(sims)
            data['NEs'] = ner_detection(data['ocr_gt'], tagger) + ner_detection(query, tagger)
            json.dump(data, open(filename_out, 'w'))
            
sns.histplot(max_sims, bins = [i / 100 for i in range(101)])
plt.savefig('tmp.png')
        

            
        
