import csv
import random

import spacy


def iterate_data(file):
    with open(file, mode='r', encoding='utf-8', newline='') as fd:
        reader = csv.reader(fd, delimiter=',')

        for row in reader:
            if len(row) == 2:
                yield row[1]


def create_data_pair(raw_data, saved_file):
    r1 = -0.15
    r2 = 0.1
    r = r1 - r2

    choose_ratio = 2
    quantity = len(raw_data)
    counter = 0
    caches = set()

    with open(saved_file, mode='a') as fd:
        for i in range(quantity):
            for _ in range(choose_ratio):
                while True:
                    j = random.choice(range(quantity))
                    if i != j and (i, j) not in caches:
                        break
                    
                caches.add((i, j))
                caches.add((j, i))
                random_similarity = r * random.random() + r2
                data = f'{raw_data[i]}\t{raw_data[j]}\t{random_similarity}\n'
                fd.write(data)
                counter += 1

                if counter % 1000 == 0:
                    print(f'[INFO] Added pair: {raw_data[i]}\t{raw_data[j]}')


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')

    # name = 'abcnews-date-text.csv'
    # name = 'cnbc_headlines.txt'
    # name = 'guardian_headlines.txt'
    # name = 'reuters_headlines.txt'
    # name = 'processed-imdb-movie-rating.csv'
    data_file = f'datasets/{name}'

    data_iter = iterate_data(data_file)
    next(data_iter, None)  # remove headers

    saved_file = f'sentence_embedding_training_data/{name}'
    saved_file = saved_file.replace('.txt', '.csv')
    with open(saved_file, mode='w') as fd:
        fd.write('sample1\tsample2\trandom_similarity\n')

    window_size = 1000
    counter = 0
    batch = []
    # max_length = 0

    while True:
        row = next(data_iter, None)

        if len(row.strip().split()) < 3 if row is not None else False:
            continue

        if row is not None:
            doc = nlp(row)
            if len([token for token in doc if token.is_oov]) > 0:
                continue

        if row is not None:
            batch.append(row)
            # if len(doc) > max_length:
            #     max_length = len(doc) 
            #     print('max_length: ', max_length, 'text: ', str(doc))
            counter += 1

        if counter >= window_size or (row is None and len(batch) > 0):
            create_data_pair(batch, saved_file)
            batch = []
            counter = 0

        if row is None:
            break
