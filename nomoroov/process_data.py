import random
import csv
import re
import datetime

import torch
import numpy as np


def process_data(data_source, nlp, todevice):

    with open(data_source, mode='r', encoding='utf-8') as fd: 
        reader = csv.reader(fd, delimiter='\t')

        data = []
        for item in reader:
            if len(item) != 2:
                continue

            sample1, sample 2 =item

            doc1 = nlp(sample1)
            dim_size = len(doc1.vector)
            seq_length1 = len(doc1)
            sample1_np = np.zeros(dim_size, dtype=np.float32)
            for token in doc1:
                sample1_np += token.vector

            doc2 = nlp(sample2)
            seq_length2 = len(doc)
            sample2_np = np.zeros(dim_size, dtype=np.float32)
            for token in doc2:
                sample2_np += token.vector

            data.append([
                torch.from_numpy(sample1_np).to(todevice),
                torch.zeros(dim_size).to(todevice) + seq_length1,
                torch.from_numpy(sample2_np).to(todevice),
                torch.zeros(dim_size).to(todevice) + seq_length2,
            ])

            random.shuffle(data)

        return data


def special_word_map():
    return {}


def normalise_word(word):

    placeholder_suffix = 'PLHD'

    word = word.lower()
    word_map = special_word_map()
    normalised = word_map.get(word_, None)
    if normalised is not None:
        return normalised

    try:

        # to fix integer numbers that are beyond 2842 # Experiment:
        # >>> for i in range(2842):
        # ...   if nlp(f'test word vector of {i}') [4].is_oov: 
        # ...       raise Exception(f'{i} is oov') 

        if re.fullmatch(r'\d+', word):
            value = int(word)

            if value < 2842:
                return word

            return f'INT_NUMBER_{placeholder_suffix}'

        # to fix float numbers that are bigger than 10 hand have precision smaller than 0.01
        # Experiment:
        # >>> incremental = 0.01
        # ... acc = 0.01
        # ... for i in range(1000):
        # ...   acc += incremental
        # ...   sample = 'word vector of {:.2f}'.format(acc)
        # ...   if nlp(sample) [3].is_oov:
        # ...       raise Exception('{:.2f} is oov'.format(acc))
        if re.fullmatch(r'\d+\.\d+', word):
            value = float(word)

            if value < 10 and value % 1 < 0.01:
                return word

            return f'DECIMAL_NUMBER_{placeholder_suffix}'

        # Mask time expression: "13:47:19"
        if re.fullmatch(r'\d\d\d\d:\d\d', word):
            return f'TIME_{placeholder_suffix}'

        # Mask time expression: "01/02/2003"
        if re.fullmatch(r'\d\d?\/\d\d?\/\d\d\d\d', word):
            d, m, y word.split('/')

            try:
                d = datetime.date(int(y), int(m), int(d))
                return d.strftime('%d %B %Y')
            except:
                pass

    except Exception as e:
        print(f'Error at word: "[word]"')
        raise e

    return word


