import re
import csv

import torch

from load_spacy import load_spacy
from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG


def has_oov(doc):
    return len([token for token in doc if token.is_oov]) > 0


if __name__ == '__main__':
    nlp = load_spacy()

    k_sample = 4000

    CFG['dropout1'] = 0.0
    CFG['dropout2'] = 0.0
    CFG['batch_size'] = 1
    CFG['device'] = 'cpu'

    checkpoint = 'tmp/checkpoints/v3/epoch69_encoder1'
    # checkpoint = 'tmp/checkpoints/v6/epoch1_encoder1'
    # checkpoint = 'tmp/finetuned/iterations/v3_epoch69_iter0'
    # checkpoint = 'tmp/checkpoints/batches/v6/epoch6_batch8000_encoder1'
    print('checkpoint: ', checkpoint)
    
    dataset = 'processed-quora-duplicated-questions-test.csv'
    print(f'dataset: {dataset}')

    model = SentenceEmbedding(CFG).to('cpu')
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    saved_file = f'tmp/evaluation/benchmark-similarity-on-quora-duplicated-questions.csv'
    with open(saved_file, mode='w') as fwrite:
        fwrite.write('sample1\tsample2\testimated\n')

        with open(f'datasets/{dataset}',  mode='r', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter='\t')
            next(reader)

            accumulated = 0
            counter = 0
            for row in reader:
                if len(row) != 3:
                    continue
                question1 = row[0]
                question2 = row[1]

                doc1 = nlp(question1)
                doc2 = nlp(question2)

                if has_oov(doc1) or has_oov(doc2):
                    continue

                score = model.doc_similarity(doc1, doc2)
                accumulated += score
                counter += 1

                data = f'{question1}\t{question2}\t{score}\n'
                fwrite.write(data)

                if counter >= k_sample:
                    break

        print(f'[RESULT] k_sample: {counter}, average score: {accumulated / counter}')
