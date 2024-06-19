import os
import csv
import argparse

import torch
from spacy.strings import hash_string
from peewee import SqliteDatabase

from cook_sentence_embedding_data import create_data_pair
from load_spacy import load_spacy
from train_sentence_embedding import CFG, FIXED_SEQUENCE_LENGTH, SIM_LOWER_R1, SIM_UPPER_R2
from fine_tune_sentence_embedding import get_database_uri, get_checkpoint_filepath
from database import BaseModel, get_model_class


def iterate_data(file):
    with open(file, mode='r', encoding='utf-8', newline='') as fd:
        reader = csv.reader(fd, delimiter='\t')

        for row in reader:
            if len(row) != 3:
                continue

            sent1, sent2 = row[0].strip(), row[1].strip()
            if len(sent1.split()) < 3 or len(sent2.split()) < 3:
                continue

            yield sent1, sent2


def estimate_similarity_fn(duplication_indexes, r1,
                           r2, model=None, propotional_threshold=0.89,
                           identical_threshold=0.95):
    def estimate_similarity(doc1, doc2):
        hashed1 = hash_string(str(doc1))
        hashed2 = hash_string(str(doc2))

        if hashed1 == hashed2:
            return (identical_threshold, 1)

        if duplication_indexes.get((hashed1, hashed2), False):
            return (propotional_threshold, 1)

        if model is None:
            return (r1, r2)

        score = model.doc_similarity(doc1, doc2)

        regulation_r1 = r1 * 1.5
        regulation_r2 = r2 * 1.5
        if regulation_r1 < score and score < regulation_r2:
            offset = regulation_r2 / 8
            return max(score - offset, -1), min(score + offset, 1)

        return (r1, r2)

    return estimate_similarity


if __name__ == '__main__':
    datasets = {
            '1': 'processed-quora-duplicated-questions-train.csv',
            '2': 'processed-quora-duplicated-questions-test.csv',
    }

    parser = argparse.ArgumentParser(
        prog='cook_quora_sentence_embeddingdata',
        description='Gererate training data from quora')
    parser.add_argument('-t', '--target',
                        choices=list(datasets.keys()),
                        required=False, default='1')
    parser.add_argument('-k', '--k_sampling', type=int, default=-5)
    parser.add_argument('-c', '--checkpoint', type=int, default=-1)
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-i', '--iteration', type=int, default=0)
    parser.add_argument('-n', '--noinference', type=int, choices=[0, 1], default=1)
    args = parser.parse_args()

    target = args.target
    k_sampling = args.k_sampling
    checkpoint_num = args.checkpoint
    current_epoch = args.epoch
    iteration = args.iteration
    no_inference = bool(args.noinference)

    nlp = load_spacy()

    if not no_inference:
        assert checkpoint_num != -1, "Must speficy model trained checkpoint."


    CFG['device'] = 'cpu'
    CFG['batch_size'] = 1

    if no_inference:
        encoder = None
    else:
        if iteration == 0 :
            checkpoint = f'tmp/checkpoints/v{checkpoint_num}/epoch{current_epoch}_encoder1'
        else:
            checkpoint = get_checkpoint_filepath(checkpoint_num, current_epoch, iteration)

        print(f'Load checkpoint: {checkpoint}')

        encoder = SentenceEmbedding(CFG).to('cpu')
        encoder.load_state_dict(torch.load(checkpoint))

    window_size = 2000
    R1 = SIM_LOWER_R1
    R2 = SIM_UPPER_R2

    name = datasets[target]
    if no_inference:
        dbfile = f'sentence_embedding_training_data/{name}.db'
    else:
        dbfile = get_database_uri(name, checkpoint_num, current_epoch, iteration)

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

    counter = 0
    batch = []
    max_length = FIXED_SEQUENCE_LENGTH
    total_quantity = 0
    duplication_indexes = {}

    while True:
        sent1, sent2 = next(data_iter, (None, None))

        if sent1 is not None and sent2 is not None:
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

        if sent1 is not None and sent2 is not None:
            batch.append(sent1)
            batch.append(sent2)
            hashed1 = hash_string(sent1)
            hashed2 = hash_string(sent2)
            duplication_indexes[(hashed1, hashed2)] = True
            counter += 2

        if counter >= window_size or (sent1 is None and len(batch) > 0):
            random_similarity = estimate_similarity_fn(duplication_indexes,
                                                       model=encoder, r1=R1,
                                                       r2=R2)
            result = create_data_pair(batch, Record, random_similarity, nlp,
                                      k_sampling, max_length,
                                      duplication_indexes)
            total_quantity += result
            batch = []
            duplication_indexes = {}
            counter = 0

        if sent1 is None or sent2 is None:
            break

    db.close()
    print(f'Done. Total records: {total_quantity}')
