import os
import csv
import argparse

import torch
from spacy.strings import hash_string
from peewee import SqliteDatabase
import msgspec

from cook_sentence_embedding_data import create_data_pair, vectorise_sequence
from load_spacy import load_spacy, get_vector_zero_index
from train_sentence_embedding import (
    CFG,
    FIXED_SEQUENCE_LENGTH,
    SIM_LOWER_R1,
    SIM_UPPER_R2,
    PROPOTIONAL_THRESHOLD
)
from fine_tune_sentence_embedding import get_database_uri, get_checkpoint_filepath
from database import BaseModel, get_model_class
from data_schema import SentencePair


def iterate_data(file):
    with open(file, mode='r', encoding='utf-8', newline='') as fd:
        reader = csv.reader(fd, delimiter='\t')

        for row in reader:
            if len(row) != 6:
                continue

            sent1, sent2, is_duplicate = row[3].strip(), row[4].strip(), row[5]
            if len(sent1.split()) < 3 or len(sent2.split()) < 3:
                continue

            yield sent1, sent2, is_duplicate


if __name__ == '__main__':
    datasets = {
            '1': 'quora-duplicate-questions-test.tsv',
    }

    parser = argparse.ArgumentParser(
        prog='cook_quora_sentence_embeddingdata',
        description='Gererate training data from quora')
    parser.add_argument('-t', '--target',
                        choices=list(datasets.keys()),
                        required=False, default='1')
    args = parser.parse_args()
    target = args.target

    nlp = load_spacy()
    vector_zero_idx = get_vector_zero_index(nlp)

    print(f'[INFO] propotional_threshold: {PROPOTIONAL_THRESHOLD}')

    name = datasets[target]
    dbfile = f'sentence_embedding_training_data/{name}.db'

    print(f'[INFO] Working on database: {dbfile}')

    if os.path.exists(dbfile):
        print(f'[INFO] Removed database {dbfile}.')
        os.remove(dbfile)

    db = SqliteDatabase(dbfile)
    Record = get_model_class(db)
    db.connect()
    db.create_tables([Record])

    data_file = f'datasets/{name}'
    data_iter = iterate_data(data_file)
    next(data_iter, None)

    print(f'[INFO] Process data: {data_file}')

    batch = []
    batch_count = 0
    batch_size = 1000
    max_length = FIXED_SEQUENCE_LENGTH
    total_quantity = 0

    for sent1, sent2, is_duplicate in data_iter:
        doc1 = nlp(sent1)
        doc2 = nlp(sent2)

        if len(doc1) > max_length or len(doc2) > max_length:
            continue

        is_skipped = False
        for token in list(doc1) + list(doc2):
            if token.is_oov:
                is_skipped = True
                break

        if is_skipped:
            continue

        if int(is_duplicate) == 1:
            r1 = PROPOTIONAL_THRESHOLD
            r2 = 1
        else:
            r1 = SIM_LOWER_R1
            r2 = SIM_UPPER_R2

        sample1 = vectorise_sequence(doc1, nlp, vector_zero_idx, max_length)
        sample2 = vectorise_sequence(doc2, nlp, vector_zero_idx, max_length)
        pair = SentencePair(
            sample1=sample1,
            sample2=sample2,
            sim_lower_r1=r1,
            sim_upper_r2=r2
        )
        batch.append({'json_data': msgspec.json.encode(pair)})
        batch_count += 1
        total_quantity += 1

        if batch_count >= batch_size:
            print('-----------------------------------------------------------')
            print(f'[INFO] Added pair: {sent1}\t{sent2}\tr1: {r1}\tr2: {r2}')
            print('-----------------------------------------------------------')

            Record.insert_many(batch).execute()
            batch = []
            batch_count = 0

    if len(batch) > 0:
        print('-----------------------------------------------------------')
        print(f'[INFO] Added pair: {sent1}\t{sent2}\tr1: {r1}\tr2: {r2}')
        print('-----------------------------------------------------------')

        Record.insert_many(batch).execute()


    db.close()
    print(f'Done. Total records: {total_quantity}')
