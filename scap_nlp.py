'''Quick proof of concept for using semantic analysis to compare CIS recommendations from
different benchmarks for similarity'''
import random
import re
import multiprocessing as mp
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import torch
from torch.functional import Tensor
from typing import Union, List
import nltk

nltk.download('punkt')
model = SentenceTransformer('stsb-roberta-large')

def encode(corpus: Union[str, List[str]]) -> Union[Tensor, List[Tensor]]:
    return model.encode(corpus, convert_to_tensor=True)

def get_similarity(corpus1, corpus2):
    if type(corpus1) == str:
        corpus1 = nltk.sent_tokenize(corpus1)
    if type(corpus2) == str:
        corpus2 = nltk.sent_tokenize(corpus2)
    return util.pytorch_cos_sim(encode(corpus1), encode(corpus2))

def get_top_k(corpus1, corpus2, k):
    top_k = []
    for sentence in list(corpus1.keys()):
        top_k.append((sentence, get_similarity(sentence, list(corpus2.keys()))[0]))
    return [(x[0], np.argpartition(-x[1], range(k))[:k]) for x in top_k]

def get_top_k_from_one(sentence, corpus, k):
    return [(sentence, np.argpartition(-get_similarity(sentence, corpus)[0], range(k))[:k])]

def compare_controls(control1, control2):
    similarities = []
    for i in ['desc', 'rationale', 'impact', 'remediation']:
        similarities.append(torch.nanmedian(get_similarity(control1[i], control2[i])))

    # Add weight to the similarity of the description
    similarities[0] = similarities[0] * 2
    return np.nanmean(similarities)

def parse_excel(file_name, sheet_name):
    return pd.read_excel(file_name, sheet_name=sheet_name, usecols='B,C,E,F,G,H')

def map_control_sentences(frame):
    control_sentences = {}
    for row in frame.itertuples(index=False):
        if not pd.isna(row[0]):
            control_sentences[row[1].strip()] = {}
            control_sentences[row[1].strip()]['number'] = row[0]
            idx = 2
            for each in ['desc', 'rationale', 'impact', 'remediation']:
                try:
                    control_sentences[row[1].strip()][each] = nltk.sent_tokenize(row[idx])
                except TypeError:
                    control_sentences[row[1].strip()][each] = [str(row[idx])]
                idx += 1
    return control_sentences

def extract_benchmark_data(filepath):
    sheet = parse_excel(filepath, 'Level 1 - Server')
    control_sentences = map_control_sentences(sheet)
    return control_sentences

def print_top_k_from_random_control(map1, map2, k):
    rand_control = list(map1.keys())[random.randrange(0, len(map1) -1)]
    top = get_top_k_from_one(rand_control, list(map2.keys()), k)
    print(f"Map1 Control Sentence: {top[0][0]}")
    print(f"Map1 Control Number: {map1[top[0][0]]}")
    print('-------------------')
    print('Most Similar:')
    for idx in top[0][1]:
        sentence = list(map2.keys())[idx]
        number = rhel8_map[sentence]
        print(f"Map 2 Control Sentence: {sentence}")
        print(f"Map 2 Control Number: {number}")
        print('***************')

def print_compare_random_control(map1, map2):
    rand7 = list(map1.keys())[random.randrange(0, len(map1) -1)]
    rand8 = list(map2.keys())[random.randrange(0, len(map2) -1)]
    print(f"Comparing \"{rand7}\" and \"{rand8}\":")
    print(f"Similarity: {compare_controls(map1[rand7], map2[rand8])}")

def print_compare_controls_by_name(map1, map2, name1, name2):
    print(f"Comparing \"{name1}\" and \"{name2}\"")
    print(f"Similarity: {compare_controls(map1[name1], map2[name2])}")

if __name__ == '__main__':
    rhel7_map = extract_benchmark_data('CIS_Red_Hat_Enterprise_Linux_7_Benchmark_v3.1.1.xlsx')
    rhel8_map = extract_benchmark_data('CIS_Red_Hat_Enterprise_Linux_8_Benchmark_v1.0.1.xlsx')
    print('############################################################')
    # print_compare_random_control(rhel7_map, rhel8_map)
    print_compare_controls_by_name(rhel7_map, rhel8_map, 'Ensure /tmp is configured', 'Ensure /tmp is configured')
    