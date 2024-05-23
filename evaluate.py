import re
import random

import torch

from load_spacy import load_spacy
from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG


if __name__ == '__main__':
    k_sample = 1000
    nlp = load_spacy()

    CFG['dropout1'] = 0.0
    CFG['dropout2'] = 0.0
    CFG['batch_size'] = 1
    CFG['device'] = 'cpu'

    checkpoint = 'tmp/checkpoints/v2/epoch7_encoder1'
    # checkpoint = 'tmp/checkpoints/batches/v2/epoch2_batch7400_encoder1'

    model = SentenceEmbedding(CFG).to('cpu')
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    dataset = 'guardian_headlines'
    saved_file = f'tmp/evaluation/benchmark-similarity-on-{dataset}.csv'
    with open(saved_file, mode='w') as fwrite:
        fwrite.write('sample1\tsample2\testimated\n')

        with open(f'datasets/{dataset}.txt',  mode='r', encoding='utf-8') as fd:
            lines = fd.readlines()
            lines = list(lines)
            data_size = len(lines)
            print('data size: ', data_size)

            for _ in range(k_sample):
                i = random.randint(0, data_size - 1)
                j = random.randint(0, data_size - 1)

                if i == j:
                    continue

                sample1 = lines[i].strip()
                sample2 = lines[j].strip()
                score = model.similarity(sample1, sample2, nlp)

                data = f'{sample1}\t{sample2}\t{score}\n'
                fwrite.write(data)
