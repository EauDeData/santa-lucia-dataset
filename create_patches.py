
import multiprocessing as mp
import json 
import argparse
import os
from multiprocessing import set_start_method

from src.datautils.create_json_scheme import process_folder

if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(
                        prog='Lorem Ipsum',
                        description='Super Trouper',
                        epilog='Uwu')

    parser.add_argument('-s', '--subset', default="gazeta-modern")
    parser.add_argument('-w', '--where_data', default="/data1tbsdd/BOE/")
    parser.add_argument('-o', '--outdir', default="data/BOE/{}/jsons")

    args = parser.parse_args()

    subset = [x for x in json.load(open('jsons/subsets.json', 'r'))["subsets"] if x["name"] == args.subset][0]
    print(f"Using {subset} subset")


    folders = [(os.path.join(args.where_data, f), args.outdir.format(f)) for f in subset["subfolders"]]
    with mp.Pool(4) as p: p.starmap(process_folder, folders)