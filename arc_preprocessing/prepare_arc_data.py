"""

This script is to be run one time
Files to generate:
1) File containing sentences from all the categories: used for w2v training
2) 3 csv files (train-val-test) with header: sentence, phrase, category
3) Subcategory to category dictionary (already have it on github, just fix the
casing of keys in that dictionary)
"""

import os
import math
import time
import json
import pickle
import pandas as pd
from tqdm import tqdm

SAVE_DIR = '../data/'

# Task 1: BEGIN
ULTIMATE_DATA_PATH = '../data/outputs/'
SPECIAL_TOKEN = '<SENT-TOKENIZED>'
SKIP_LEN = len(SPECIAL_TOKEN)


def make_ultimate_sentences():
    if os.path.exists(SAVE_DIR + 'arc-sentences-all.txt'):
        print('arc-sentences-all.txt already exists'.
              format(SAVE_DIR))
        return
    start = time.clock()
    all_sentences = ""
    for file_name in tqdm(os.listdir(ULTIMATE_DATA_PATH), desc='Files parsed'):
        with open(ULTIMATE_DATA_PATH + file_name, 'r', errors='ignore') as data_file:
            data = data_file.readlines()
            for line in tqdm(data, desc='Lines parsed in {}'.format(file_name)):
                if SPECIAL_TOKEN in line:
                    all_sentences += line[SKIP_LEN:]

    with open(SAVE_DIR + 'arc-sentences-all.txt', 'w') as sent_file:
        sent_file.write(all_sentences)
    print("\nUltimate sentence file prepared in {:.4f} seconds.".
          format(time.clock() - start))


# Task 2: BEGIN
RWS_TUPLE_FILE = '../data/ALL-productID-sent-RWS_tuple'


def make_csv_datafiles():
    if os.path.exists(SAVE_DIR + "df_arc_train.csv") \
        or os.path.exists(SAVE_DIR + "df_arc_val.csv") \
            or os.path.exists(SAVE_DIR + "df_arc_test.csv"):
        print('The train-val-test csvs already exist for ARC experiments')
        return
    start = time.clock()
    # use latin1 encoding for reading csv or else face UnicodeDecodeError
    all_rws = pd.read_csv(RWS_TUPLE_FILE, names=['ProductID', 'ProductSent',
                          'ProductPhrase', 'Category'], encoding='latin1')
    # shuffle
    all_rws = all_rws.sample(frac=1).reset_index(drop=True)
    total_rows = len(all_rws)
    # train:val:test <=> 70:10:20 split
    trn_idx = math.floor(0.7 * total_rows)
    val_idx = trn_idx + math.floor(0.1 * total_rows)
    all_rws[:trn_idx].to_csv(SAVE_DIR + "df_arc_train.csv", index=False)
    all_rws[trn_idx:val_idx].to_csv(SAVE_DIR + "df_arc_val.csv", index=False)
    all_rws[val_idx:].to_csv(SAVE_DIR + "df_arc_test.csv", index=False)
    print("train-val-test csvs for ARC created in {:.4f} seconds".
          format(time.clock() - start))


# Task 3: BEGIN
BAD_DICT_FILE = '../data/subCat2CatMap.pkl'


def fix_dict_key_casing():
    if os.path.exists(SAVE_DIR + 'subCat2CatMap.json'):
        print('subCat2CatMap alredy exists')
        return
    start = time.clock()
    with open(BAD_DICT_FILE, 'rb') as f:
        bad_map = pickle.load(f)
    good_map = {}
    for sub_cat in bad_map.keys():
        good_map[sub_cat.lower().replace('&', 'and')] = \
            bad_map[sub_cat].replace('_', ' ')
    with open(SAVE_DIR + 'subCat2CatMap.json', 'w') as g:
        json.dump(good_map, g)
    print('Fixed keys in the subCat2CatMap in {:.4f} seconds'.
          format(time.clock() - start))


if __name__ == '__main__':
    start = time.clock()
    make_ultimate_sentences()
    make_csv_datafiles()
    fix_dict_key_casing()
    print("\nDone preparing data for ARC experiments in {:.4f} seconds.".
          format(time.clock() - start))
