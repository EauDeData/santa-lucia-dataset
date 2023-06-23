
import multiprocessing as mp
import json 
import argparse
import os
import layoutparser as lp
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from src.datautils.create_json_scheme import just_save_numpy


if __name__ == '__main__':
    mp.set_start_method('spawn')


    parser = argparse.ArgumentParser(
                        prog='Lorem Ipsum',
                        description='Super Trouper',
                        epilog='Uwu')

    parser.add_argument('-s', '--subset', default="gazeta-modern")
    parser.add_argument('-w', '--where_data', default="/data1tbsdd/BOE/")
    parser.add_argument('-t', '--threads', default=2, type=int)


    args = parser.parse_args()
    
    subset = [x for x in json.load(open('jsons/subsets.json', 'r'))["subsets"] if x["name"] == args.subset][0]
    folders = [(os.path.join(args.where_data, f), args.threads) for f in subset["subfolders"]]
    for folder in folders[::-1]:
        just_save_numpy(*folder)

