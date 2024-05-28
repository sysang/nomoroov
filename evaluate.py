import re
import random

import torch

from load_spacy import load_spacy
from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG


if __name__ == '__main__':
    k_sample = 4000
    nlp = load_spacy()

    CFG['dropout1'] = 0.0
    CFG['dropout2'] = 0.0
    CFG['batch_size'] = 1
    CFG['device'] = 'cpu'

    checkpoint = 'tmp/checkpoints/v3/epoch40_encoder1'
    # checkpoint = 'tmp/checkpoints/batches/v2/epoch2_batch7400_encoder1'
    print('checkpoint: ', checkpoint)

    model = SentenceEmbedding(CFG).to('cpu')
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    dataset1 = 'cnbc_headlines'
    dataset2 = 'guardian_headlines'

    with open(f'datasets/{dataset1}.txt',  mode='r', encoding='utf-8') as fd:
        samples_1 = list(fd.readlines())
        samples_size_1 = len(samples_1)

    with open(f'datasets/{dataset2}.txt',  mode='r', encoding='utf-8') as fd:
        samples_2 = list(fd.readlines())
        samples_size_2 = len(samples_2)

    print(f'{dataset1}: {samples_size_1}, {dataset2}: {samples_size_2}')

    saved_file = f'tmp/evaluation/benchmark-{dataset1}-{dataset2}.csv'
    with open(saved_file, mode='w') as fwrite:
        fwrite.write('sample1\tsample2\testimated\n')

        accumulated = 0
        for _ in range(k_sample):
            i = random.randint(0, samples_size_1 - 1)
            sample1 = samples_1[i].strip()

            j = random.randint(0, samples_size_2 - 1)
            sample2 = samples_2[j].strip()

            score = model.similarity(sample1, sample2, nlp)
            accumulated += score

            data = f'{sample1}\t{sample2}\t{score}\n'
            fwrite.write(data)

    print(f'[RESULT] k_sample: {k_sample}, average score: {(accumulated / k_sample):0.5f}')
