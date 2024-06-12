import torch

from sentence_embedding_model import SentenceEmbedding
from sentence_embedding_model_v7 import SentenceEmbeddingV7
from train_sentence_embedding import CFG
from load_spacy import load_spacy


# model = SentenceEmbedding(CFG, finetuning=False).to('cpu')
# print('number of parameters: ', sum( p.numel() for p in model.parameters() if p.requires_grad))

# model = SentenceEmbedding(CFG, finetuning=True).to('cpu')
# print('number of parameters: ', sum( p.numel() for p in model.parameters() if p.requires_grad))

model1 = SentenceEmbeddingV7(finetuning=False).to('cpu')
print('number of parameters: ', sum( p.numel() for p in model1.parameters() if p.requires_grad))

model2 = SentenceEmbeddingV7(finetuning=True).to('cpu')
model2.eval()
print('number of parameters: ', sum( p.numel() for p in model2.parameters() if p.requires_grad))
