import spacy
import torch
import numpy as np


VECTOR_ZERO_SYMBOL = 'vector_zero'


def load_spacy(disable=['tagger', 'ner', 'parser', 'attribute_ruler', 'lemmatizer']):
    nlp = spacy.load('en_core_web_lg', disable=disable)
    nlp.vocab.vectors.resize((nlp.vocab.vectors.shape[0] + 1, nlp.vocab.vectors.shape[1]))
    item_id = nlp.vocab.strings.add(VECTOR_ZERO_SYMBOL)
    nlp.vocab.vectors.add(item_id, vector=np.zeros(300, dtype=np.float32))

    return nlp


def get_vector_zero_index(nlp):
    vector_zero_key = nlp.vocab.strings[VECTOR_ZERO_SYMBOL]
    vector_zero_idx = nlp.vocab.vectors.find(key=vector_zero_key)
    return vector_zero_idx


def translate_sequence(sequence, nlp):
    vector_zero_idx = get_vector_zero_index(nlp)
    tokens = [ nlp.vocab.strings[nlp.vocab.vectors.find(row=idx)] for idx in sequence if idx != vector_zero_idx]
    return ' '.join(tokens)
