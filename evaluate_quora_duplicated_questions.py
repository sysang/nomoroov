import re
import csv

import torch

from load_spacy import load_spacy
from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG


if __name__ == '__main__':
    nlp = load_spacy()

    CFG['dropout1'] = 0.0
    CFG['dropout2'] = 0.0
    CFG['batch_size'] = 1
    CFG['device'] = 'cpu'

    checkpoint = 'tmp/checkpoints/v3/epoch10_encoder1'
    # checkpoint = 'tmp/checkpoints/batches/v2/epoch2_batch7400_encoder1'
    print('checkpoint: ', checkpoint)
    
    dataset = 'datasets/quora-duplicate-questions.tsv'
    print('dataset: ', dataset)

    model = SentenceEmbedding(CFG).to('cpu')
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    saved_file = f'tmp/evaluation/benchmark-similarity-on-quora-duplicated-questions.csv'
    with open(saved_file, mode='w') as fwrite:
        fwrite.write('sample1\tsample2\testimated\n')

        with open(dataset,  mode='r', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter='\t')
            next(reader)

            accumulated = 0
            counter = 0
            for row in reader:
                if len(row) != 6:
                    continue
                question1 = row[3]
                question2 = row[4]
                is_duplicated = row[5]

                if is_duplicated == '0':
                    continue

                # doc1 = nlp(question1)
                # doc2 = nlp(question2)

                # if has_oov(doc1) or has_oov(doc2):
                #     continue

                score = model.similarity(question1, question2, nlp)
                accumulated += score
                counter += 1

                data = f'{question1}\t{question2}\t{score}\n'
                fwrite.write(data)

        print(f'Done. Average score: {accumulated / counter}')
