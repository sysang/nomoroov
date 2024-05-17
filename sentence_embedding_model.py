from torch import nn
from torch.autograd import Variable
import torch


DEFAULT_CFG = {
    'embed_size': 300,
    'hidden_size1': 30,
    'hidden_size2': 75,
    'dropout1': 0.23,
    'dropout2': 0.97,
    'num_layers1': 2,
    'num_layers2': 4,
    'device': 'cpu',
    'batch_size': 1024
}


class SentenceEmbedding(nn.Module):
    def __init__(self, config=DEFAULT_CFG, batch_size=None, dropout1=None, dropout2=None):
        super(SentenceEmbedding, self).__init__()
        self.embed_size = config['embed_size']
        self.hidden_size1 = config['hidden_size1']
        self.hidden_size2 = config['hidden_size2']
        self.dropout1 = config['dropout1']
        self.dropout2 = config['dropout2']
        self.num_layers1 = config['num_layers1']
        self.num_layers2 = config['num_layers2']
        self.batch_size = batch_size if batch_size is not None else config['batch_size']
        self.device = config['device']

        self.compress1 = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size1)
        self.decode1 = nn.GRU(input_size=self.hidden_size1, hidden_size=self.hidden_size1,
                              num_layers=self.num_layers1, dropout=self.dropout1)
        self.decode2 = nn.GRU(input_size=self.hidden_size1, hidden_size=self.hidden_size2,
                              num_layers=self.num_layers2, dropout=self.dropout2)

    def initHiddenCell(self, batch_size, hidden_size, num_layers=1):
        return Variable(torch.rand(num_layers, batch_size, hidden_size)).to(self.device).mul(-1000)

    def forward(self, input):
        hidden1 = self.initHiddenCell(self.batch_size, self.hidden_size1)
        embedded1, hidden1 = self.compress1(input, hidden1)

        hidden2 = self.initHiddenCell(self.batch_size, self.hidden_size1)
        hidden2 = torch.cat((hidden1, hidden2), dim=0)
        embedded2, _ = self.decode1(embedded1, hidden2)

        hidden3 = self.initHiddenCell(self.batch_size, self.hidden_size2, self.num_layers2)
        embedded3, _ = self.decode2(embedded2, hidden3)

        return embedded3[-1]
