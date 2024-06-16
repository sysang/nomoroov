import re
import csv

import torch
from torch import nn
import numpy as np

from load_spacy import load_spacy
from sentence_embedding_model import SentenceEmbedding
from sentence_embedding_model_v3 import SentenceEmbeddingV3
from sentence_embedding_model_v7 import SentenceEmbeddingV7
from train_sentence_embedding import get_word_vector_matrix, CFG
from cook_sentence_embedding_data import vectorise_sequence


def has_oov(doc):
    return len([token for token in doc if token.is_oov]) > 0


if __name__ == '__main__':
    nlp = load_spacy()
    vector_zero_key = nlp.vocab.strings["vector_zero"]
    vector_zero_idx = nlp.vocab.vectors.find(key=vector_zero_key)

    k_sample = 4000

    # CFG['dropout1'] = 0.0
    # CFG['dropout2'] = 0.0
    CFG['batch_size'] = 1
    CFG['device'] = 'cpu'

    # checkpoint = 'tmp/checkpoints/v3/epoch69_encoder1'
    # checkpoint = 'tmp/checkpoints/v8/epoch2_encoder2'
    # checkpoint = 'tmp/finetuned/iterations/v3_epoch69_iter0'
    checkpoint = 'tmp/checkpoints/batches/v9/epoch1_batch1800_encoder1'
    print('checkpoint: ', checkpoint)
    
    dataset = 'processed-quora-duplicated-questions-test.csv'
    print(f'dataset: {dataset}')

    word_embedding = nn.Embedding.from_pretrained(get_word_vector_matrix(nlp, 'cpu'))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # model = SentenceEmbedding(CFG).to('cpu')
    model = SentenceEmbeddingV7(CFG, inferring=True, batch_size=1).to('cpu')
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

                if len(doc1) > 40 or len(doc2) > 40:
                    continue

                sample1 = vectorise_sequence(doc1, nlp, vector_zero_idx, 40)
                sample1 = torch.from_numpy(np.array(sample1))
                sample1 = word_embedding(sample1)
                sample1 = sample1.unsqueeze(dim=1)
                embedded1 = model(sample1)

                sample2 = vectorise_sequence(doc2, nlp, vector_zero_idx, 40)
                sample2 = torch.from_numpy(np.array(sample2))
                sample2 = word_embedding(sample2)
                sample2 = sample2.unsqueeze(dim=1)
                embedded2 = model(sample2)

                score = cos(embedded1, embedded2)
                accumulated += score
                counter += 1

                data = f'{question1}\t{question2}\t{score}\n'
                fwrite.write(data)

                if counter >= k_sample:
                    break

        print(f'[RESULT] k_sample: {counter}, average score: {accumulated.item() / counter}')
