import spacy
import torch
import numpy as np


def load_spacy(disable=['tagger', 'ner', 'parser', 'attribute_ruler', 'lemmatizer']):
    nlp = spacy.load('en_core_web_lg', disable=disable)
    nlp.vocab.vectors.resize((nlp.vocab.vectors.shape[0] + 1, nlp.vocab.vectors.shape[1]))
    item_id = nlp.vocab.strings.add("vector_zero")
    nlp.vocab.vectors.add(item_id, vector=np.zeros(300, dtype=np.float32))

    return nlp
