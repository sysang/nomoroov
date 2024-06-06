import torch

from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import CFG
from load_spacy import load_spacy


model = SentenceEmbedding(CFG, finetuning=False).to('cpu')
print('number of parameters: ', sum( p.numel() for p in model.parameters() if p.requires_grad))

model = SentenceEmbedding(CFG, finetuning=True).to('cpu')
print('number of parameters: ', sum( p.numel() for p in model.parameters() if p.requires_grad))
