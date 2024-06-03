import re
import random
import csv
import math

import torch

from load_spacy import load_spacy
from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG


def has_oov(doc):
    return len([token for token in doc if token.is_oov]) > 0


if __name__ == '__main__':
    k_sample = 4000
    nlp = load_spacy()

    CFG['dropout1'] = 0.0
    CFG['dropout2'] = 0.0
    CFG['batch_size'] = 1
    CFG['device'] = 'cpu'

    # checkpoint = 'tmp/checkpoints/v3/epoch77_encoder1'
    checkpoint = 'tmp/checkpoints/v5/epoch1_encoder1'
    # checkpoint = 'tmp/finetuned/iterations/v3_epoch69_iter0'
    
    dataset = 'processed-quora-duplicated-questions-test.csv'
    print('checkpoint: ', checkpoint)

    model = SentenceEmbedding(CFG).to('cpu')
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    # dataset1 = 'cnbc_headlines'
    # dataset2 = 'guardian_headlines'

    dataset = 'processed-quora-duplicated-questions-test.csv'
    with open(f'datasets/{dataset}',  mode='r', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter='\t')
        next(reader);
        samples = [row[0] for row in reader]
        middle = int(len(samples) / 2)

    samples_1 = samples[0:middle]
    samples_size_1 = len(samples_1)
    samples_2 = samples[middle:]
    samples_size_2 = len(samples_2)

    # with open(f'datasets/{dataset1}.txt',  mode='r', encoding='utf-8') as fd:
    #     samples_1 = list(fd.readlines())
    #     samples_size_1 = len(samples_1)

    # with open(f'datasets/{dataset2}.txt',  mode='r', encoding='utf-8') as fd:
    #     samples_2 = list(fd.readlines())
    #     samples_size_2 = len(samples_2)

    print(f'dataset: datasets/{dataset}')
    print(f'first part: {samples_size_1}, second part: {samples_size_2}')

    saved_file = f'tmp/evaluation/benchmark-{dataset}.csv'
    with open(saved_file, mode='w') as fwrite:
        fwrite.write('sample1\tsample2\testimated\n')

        accumulated = 0
        counter = 0
        for _ in range(k_sample):
            i = random.randint(0, samples_size_1 - 1)
            sample1 = samples_1[i].strip()

            j = random.randint(0, samples_size_2 - 1)
            sample2 = samples_2[j].strip()

            doc1 = nlp(sample1)
            doc2 = nlp(sample2)

            if has_oov(doc1) or has_oov(doc2):
                continue

            score = model.doc_similarity(doc1, doc2)
            accumulated += score
            counter += 1

            data = f'{sample1}\t{sample2}\t{score}\n'
            fwrite.write(data)

    print(f'[RESULT] k_sample: {counter}, average score: {(accumulated / counter):0.5f}')
