from torch import nn
from torch.autograd import Variable
import torch

import numpy as np


DEFAULT_CFG = {
    'embed_size': 300,
    'hidden_size1': 16,   # v2
    'hidden_size2': 64,  # v2
    'dropout1': 0.17,
    'dropout2': 0.81,
    'num_layers1': 2,
    'num_layers2': 3,     # v2
    'device': 'cpu',
    'batch_size': 1
}


class SentenceEmbeddingV3(nn.Module):
    def __init__(self, config=DEFAULT_CFG, batch_size=None, dropout1=None, dropout2=None, device=None):
        super(SentenceEmbeddingV3, self).__init__()
        self.embed_size = config['embed_size']
        self.hidden_size1 = config['hidden_size1']
        self.hidden_size2 = config['hidden_size2']
        self.dropout1 = dropout1 if dropout1 is not None else config['dropout1']
        self.dropout2 = dropout2 if dropout2 is not None else config['dropout2']
        self.num_layers1 = config['num_layers1']
        self.num_layers2 = config['num_layers2']
        self.batch_size = batch_size if batch_size is not None else config['batch_size']
        self.device = device if device is not None else config['device']

        self.compress1 = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size1,
                                num_layers=self.num_layers1)
        self.decode1 = nn.GRU(input_size=self.hidden_size1, hidden_size=self.hidden_size1,
                              num_layers=self.num_layers1, dropout=self.dropout1)
        self.decode2 = nn.GRU(input_size=self.hidden_size1, hidden_size=self.hidden_size2,
                              num_layers=self.num_layers2, dropout=self.dropout2)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def initHiddenCell(self, batch_size, hidden_size, num_layers=1):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size)).to(self.device)

    def forward(self, input):
        hidden1 = self.initHiddenCell(self.batch_size, self.hidden_size1, 2)
        embedded1, hidden1 = self.compress1(input, hidden1)

        hidden2 = self.initHiddenCell(self.batch_size, self.hidden_size1)
        # hidden2 = torch.cat((hidden1, hidden2), dim=0)
        embedded2, _ = self.decode1(embedded1, hidden1)

        hidden3 = self.initHiddenCell(self.batch_size, self.hidden_size2, self.num_layers2)
        _, hidden3 = self.decode2(embedded2, hidden3)

        return torch.cat((hidden3[-1], hidden3[-2]), dim=1)

    def tensorise(self, doc):
        tensor = torch.zeros(len(doc), len(doc[0].vector), dtype=torch.float32)

        for idx, token in enumerate(doc):
            tensor[idx] = torch.from_numpy(token.vector)

        tensor = tensor.to(self.device).unsqueeze(1)

        return tensor

    def text_similarity(self, text1, text2, nlp):
        v1 = self.tensorise(text1, nlp(text1))
        v2 = self.tensorise(text2, nlp(text2))

        return self.similarity(v1, v2)

    def doc_similarity(self, doc1, doc2):
        v1 = self.tensorise(doc1)
        v2 = self.tensorise(doc2)

        return self.similarity(v1, v2)

    def similarity(self, v1, v2):
        embedded1 = self.forward(v1)
        embedded2 = self.forward(v2)

        distance = self.cos(embedded1, embedded2)

        return distance[0].item()
