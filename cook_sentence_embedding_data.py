import os
import random
import argparse

import torch
import msgspec

from spacy.strings import hash_string
from torch import nn
from peewee import SqliteDatabase

from train_sentence_embedding import CFG, FIXED_SEQUENCE_LENGTH, SIM_LOWER_R1, SIM_UPPER_R2, IDENTICAL_THRESHOLD
from database import BaseModel
from data_schema import SentencePair
from load_spacy import load_spacy


cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def iterate_data(file):
    with open(file, mode='r', encoding='utf-8', newline='') as fd:
        lines = fd.readlines()

        for line in lines:
            yield line.strip()


def random_similarity_fn(r1=-0.125, r2=0.125, identical_threshold=0.95):
    def random_similarity(doc1, doc2):
        text1 = str(doc1)
        text2 = str(doc2)

        if text1 == text2:
            return (identical_threshold, 1)

        return (r1, r2)

    return random_similarity


def vectorise_sequence(doc, nlp, vector_zero_idx, fixed_sequence_length):
    sequence = []
    for token in doc:
        if not token.text.strip():
            continue

        idx = nlp.vocab.vectors.find(key=token.lex.orth)

        if idx == -1:
            print(f'[WARNING] skipped invalid data: {str(sample)}, orth: \
{[(token.text, token.lex.orth) for token in nlp(sample)]}')
            return None
        else:
            sequence.append(idx)

    length = len(sequence)
    padlen = fixed_sequence_length - length
    if padlen >= 0:
        padding = [vector_zero_idx for _ in range(padlen)]
        sequence = sequence + padding
    else:
        sequence = sequence[0:fixed_sequence_length]

    return sequence


def create_data_pair(raw_data, Record, estimate_similarity, nlp, k_sampling,
                     fixed_sequence_length, duplication_indexes=None):
    vector_zero_key = nlp.vocab.strings["vector_zero"]
    vector_zero_idx = nlp.vocab.vectors.find(key=vector_zero_key)

    quantity = len(raw_data)
    counter = 0
    caches = set()
    batch = []
    batch_size = 1000
    batch_count = 0

    for i in range(quantity):
        text11 = raw_data[i]
        doc11 = nlp(text11)
        text12 = text11
        doc12 = nlp(text12)
        sample11 = vectorise_sequence(
            doc11, nlp, vector_zero_idx, fixed_sequence_length)
        sample12 = sample11

        if sample11 is None:
            break

        r11, r12 = estimate_similarity(doc1=doc11, doc2=doc12)
        pair = SentencePair(
            sample1=sample11,
            sample2=sample12,
            sim_lower_r1=r11,
            sim_upper_r2=r12
        )
        batch.append({'json_data': msgspec.json.encode(pair)})
        counter += 1
        batch_count += 1

        text22 = 'n/a'
        r21 = 'n/a'
        r22 = 'n/a'
        if duplication_indexes is not None and i + 1 <= quantity - 1:
            hashed11 = hash_string(text11)
            j = i + 1
            text22 = raw_data[j]
            doc22 = nlp(text22)
            hashed22 = hash_string(text22)

            if duplication_indexes.get((hashed11, hashed22), None) is not None:
                caches.add((i, j))
                caches.add((j, i))

                sample22 = vectorise_sequence(
                    doc22, nlp, vector_zero_idx, fixed_sequence_length)

                r21, r22 = estimate_similarity(doc1=doc11, doc2=doc22)
                if sample22 is not None:
                    pair = SentencePair(
                        sample1=sample11,
                        sample2=sample22,
                        sim_lower_r1=r21,
                        sim_upper_r2=r22
                    )
                    batch.append({'json_data': msgspec.json.encode(pair)})
                    counter += 1
                    batch_count += 1

        sampling_count = 0
        while True:
            while True:
                j = random.choice(range(quantity))
                if i !=j and (i, j) not in caches:
                    break

            caches.add((i, j))
            caches.add((j, i))
            text32 = raw_data[j]
            doc32 = nlp(text32)

            sample32 = vectorise_sequence(
                doc32, nlp, vector_zero_idx, fixed_sequence_length)

            if sample32 is not None:
                r31, r32 = estimate_similarity(doc1=doc11, doc2=doc32)
                pair = SentencePair(
                    sample1=sample11,
                    sample2=sample32,
                    sim_lower_r1=r31,
                    sim_upper_r2=r32
                )
                batch.append({'json_data': msgspec.json.encode(pair)})

                sampling_count +=1
                batch_count += 1
                counter += 1

            if sampling_count >= k_sampling:
                break

        if batch_count >= batch_size:
            print('-----------------------------------------------------------')
            print(f'[INFO] Added pair: {text11}\t{text12}\tr1: {r11}\tr2: {r12}')
            print(f'[INFO] Added pair: {text11}\t{text22}\tr1: {r21}\tr2: {r22}')
            print(f'[INFO] Added pair: {text11}\t{text32}\tr1: {r31}\tr2: {r32}')
            print('-----------------------------------------------------------')
            Record.insert_many(batch).execute()
            batch = []
            batch_count = 0

    if len(batch) > 0:
        print(f'[INFO] Added pair: {text11}\t{text12}\tr1: {r11}\tr2: {r12}')
        print(f'[INFO] Added pair: {text11}\t{text22}\tr1: {r21}\tr2: {r22}')
        print(f'[INFO] Added pair: {text11}\t{text32}\tr1: {r31}\tr2: {r32}')
        Record.insert_many(batch).execute()

    return counter


if __name__ == '__main__':
    datasets = {
            '1': 'wikidata-text-part-0.txt',
            '2': 'wikidata-text-part-1.txt',
            '3': 'wikidata-text-part-2.txt',
            '4': 'wikidata-text-part-3.txt',
            '5': 'wikidata-text-part-4.txt',
            '6': 'wikidata-text-part-5.txt',
            '7': 'abcnews-date-text-part-0.txt',
            '8': 'abcnews-date-text-part-1.txt',
            '9': 'processed-imdb-movie-rating-part-0.txt',
            '10': 'processed-imdb-movie-rating-part-1.txt',
            '11': 'cnbc_headlines.txt',  # skipped
            '12': 'reuters_headlines.txt',
            '13': 'guardian_headlines.txt',  # skipped
            '14': 'gutenberg-project-book-part-0.txt',
            '15': 'gutenberg-project-book-part-1.txt',
            '16': 'gutenberg-project-book-part-2.txt',
            '-1': 'samples.txt',
    }

    parser = argparse.ArgumentParser(prog='cook_sentence_embedding_data',
                                     description='Gererate training data')
    parser.add_argument('-t', '--target',
                        choices=list(datasets.keys()),
                        required=False, default='11')

    args = parser.parse_args()
    target = args.target
    name = datasets[target]

    nlp = load_spacy()

    WINDOW_SIZE = 2500
    K_SAMPLING = 20
    R1 = SIM_LOWER_R1
    R2 = SIM_UPPER_R2

    random_similarity = random_similarity_fn(r1=R1, r2=R2, identical_threshold=IDENTICAL_THRESHOLD)

    print(f'[INFO] Process dataset: {name}')

    dbfile = f'sentence_embedding_training_data/{name}.db'
    if os.path.exists(dbfile):
        print(f'[INFO] Removed database {dbfile}.')
        os.remove(dbfile)

    db = SqliteDatabase(dbfile)

    class Record(BaseModel):
        table_name = 'record'
        class Meta:
            database = db

    db.connect()
    db.create_tables([Record])

    data_file = f'datasets/{name}'
    data_iter = iterate_data(data_file)

    counter = 0
    batch = []
    max_length = 40
    total_quantity = 0

    while True:
        row = next(data_iter, None)

        if len(row.strip().split()) < 3 if row is not None else False:
            continue

        if row is not None:
            doc = nlp(row)

            if len(doc) > max_length:
                continue

            is_skipped = False
            for token in doc:
                if token.is_oov:
                    is_skipped = True
                    break

            if is_skipped:
                continue

        if row is not None:
            batch.append(row)
            counter += 1

        if counter >= WINDOW_SIZE or (row is None and len(batch) > 0):
            result = create_data_pair(batch, Record, random_similarity, nlp,
                                      K_SAMPLING, FIXED_SEQUENCE_LENGTH)
            total_quantity += result
            batch = []
            counter = 0

        if row is None:
            break

    db.close()
    print(f'Done. Total records: {total_quantity}')
