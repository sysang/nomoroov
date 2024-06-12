from torch import nn
from torch.autograd import Variable
import torch


DEFAULT_CFG = {
    'embed_size': 300,
    'compress_size1': 60,
    'hidden_size1': 16,
    'hidden_size2': 64,
    'dropout': 0.61,
    'num_layers1': 4,
    'num_layers2': 5,
    'device': 'cpu',
    'batch_size': 1,
    'fixed_sequence_length': 40,
    'token_noise_magnitue': 2.9,
    'sequence_noise_ratio': 0.33
}


class SentenceEmbeddingV7(nn.Module):
    def __init__(
            self,
            config=DEFAULT_CFG,
            batch_size=None,
            dropout=None,
            device=None,
            finetuning=False,
            inferring=False):
        super(SentenceEmbeddingV7, self).__init__()
        self.embed_size = config['embed_size']
        self.compress_size1 = config['compress_size1']
        self.hidden_size1 = config['hidden_size1']
        self.hidden_size2 = config['hidden_size2']
        self.dropout = dropout if dropout is not None else config['dropout']
        self.num_layers1 = config['num_layers1']
        self.num_layers2 = config['num_layers2']
        self.batch_size = batch_size if batch_size is not None else config['batch_size']
        self.device = device if device is not None else config['device']
        self.fixed_sequence_length = config['fixed_sequence_length']
        self.finetuning = finetuning
        self.inferring = inferring

        self.num_layers_i = 2

        self.compress1 = nn.Linear(self.embed_size, 170, bias=False)
        self.compress2 = nn.Linear(170, 100, bias=False)
        self.compress3 = nn.Linear(100, self.compress_size1, bias=False)

        self.encoder1 = nn.GRU(input_size=self.compress_size1, hidden_size=self.hidden_size1,
                               num_layers=self.num_layers1)
        # self.decode1 = nn.GRU(input_size=self.hidden_size1, hidden_size=self.hidden_size1,
        #                       num_layers=1)
        self.decode2 = nn.GRU(input_size=self.hidden_size1, hidden_size=self.hidden_size2,
                              num_layers=self.num_layers2, dropout=self.dropout)

        self.linear_out1 = nn.Linear(self.hidden_size2, self.hidden_size2, bias=False)
        self.linear_out2 = nn.Linear(self.hidden_size2, self.hidden_size2, bias=False)
        self.linear_out3 = nn.Linear(self.hidden_size2, self.hidden_size2, bias=False)
        self.linear_out4 = nn.Linear(self.hidden_size2, self.hidden_size2, bias=False)

        self.norm = nn.BatchNorm1d(self.hidden_size2 * 5)

        if finetuning or inferring:
            for p in self.compress1.parameters():
                p.requires_grad = False
            for p in self.compress2.parameters():
                p.requires_grad = False
            for p in self.compress3.parameters():
                p.requires_grad = False
            for p in self.encoder1.parameters():
                p.requires_grad = False
            # for p in self.decode1.parameters():
            #     p.requires_grad = False

        if inferring:
            self.compress_wv = self.compress_flexible_sequence
            self.apply_noise = self.bypass_applying_noise

            for p in self.decode2.parameters():
                p.requires_grad = False
            for p in self.linear_out1.parameters():
                p.requires_grad = False
            for p in self.linear_out2.parameters():
                p.requires_grad = False
            for p in self.linear_out3.parameters():
                p.requires_grad = False
            for p in self.linear_out4.parameters():
                p.requires_grad = False
        else:
            self.compress_wv = self.compress_fixed_sequence

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.token_noise_magnitue = torch.tensor(
            config['token_noise_magnitue'], dtype=torch.float32).to(device)
        self.sequence_noise_ratio = torch.tensor(
            config['sequence_noise_ratio'], dtype=torch.float32).to(device)

        self.threshold = torch.ones(
            (self.fixed_sequence_length, self.batch_size, 1),
            dtype=torch.float32).to(self.device).mul(self.sequence_noise_ratio)

    def initHiddenCell(self, batch_size, hidden_size, num_layers=1):
        return Variable(torch.ones(num_layers, batch_size, hidden_size)).to(self.device)

    def forward(self, input):
        compressed = self.compress_wv(input)

        encoded_hidden_1 = self.initHiddenCell(self.batch_size, self.hidden_size1,
                                               self.num_layers1)
        encoded_1, encoded_hidden_1 = self.encoder1(compressed, encoded_hidden_1)

        sample_length_filter = input[:, :, 0:self.hidden_size1] != torch.zeros(
            encoded_1.shape, dtype=torch.float32).to(self.device)
        sample_length_filter = sample_length_filter.int()
        filtered_encoded_1 = encoded_1.mul(sample_length_filter)

        noised_encoded_1 = self.apply_noise(filtered_encoded_1)

        # decoded_hidden_1 = self.initHiddenCell(self.batch_size, self.hidden_size1,
        #                                        self.num_layers1)
        # decoded_1, _ = self.decode1(noised_encoded_1, decoded_1)

        decoded_hidden_2 = self.initHiddenCell(self.batch_size, self.hidden_size2,
                                               self.num_layers2)
        _, hidden3 = self.decode2(noised_encoded_1, decoded_hidden_2)

        output1 = self.linear_out1(hidden3[0])
        output2 = self.linear_out2(hidden3[1])
        output3 = self.linear_out3(hidden3[2])
        output4 = self.linear_out4(hidden3[3])
        output5 = hidden3[4]

        output = torch.cat((output1, output2, output3, output4, output5), dim=1)

        return self.norm(output)

    def apply_noise(self, sample):
        fixed_sequence_length, batch_size, embed_size = sample.shape

        sample_length_filter = sample != torch.zeros(
            sample.shape, dtype=torch.float32).to(self.device)
        sample_length_filter = sample_length_filter.int()

        noise_masking = torch.rand(
            (fixed_sequence_length, batch_size, 1), dtype=torch.float32).to(self.device)
        noise_masking = noise_masking < self.threshold
        noise_masking = noise_masking.int()

        noise = torch.rand(sample.shape, dtype=torch.float32).to(self.device)
        noise = noise.mul(self.token_noise_magnitue).mul(sample)
        noise = noise.mul(noise_masking).mul(sample_length_filter)
        noised = sample.add(noise)

        return noised

    def bypass_applying_noise(self, sample):
        return sample

    def compress_linear(self, input):
        return self.compress3(self.compress2(self.compress1(input)))

    def compress_fixed_sequence(self, input):
        return torch.stack((
                self.compress_linear(input[0]),
                self.compress_linear(input[1]),
                self.compress_linear(input[2]),
                self.compress_linear(input[3]),
                self.compress_linear(input[4]),
                self.compress_linear(input[5]),
                self.compress_linear(input[6]),
                self.compress_linear(input[7]),
                self.compress_linear(input[8]),
                self.compress_linear(input[9]),
                self.compress_linear(input[10]),
                self.compress_linear(input[11]),
                self.compress_linear(input[12]),
                self.compress_linear(input[13]),
                self.compress_linear(input[14]),
                self.compress_linear(input[15]),
                self.compress_linear(input[16]),
                self.compress_linear(input[17]),
                self.compress_linear(input[18]),
                self.compress_linear(input[19]),
                self.compress_linear(input[20]),
                self.compress_linear(input[21]),
                self.compress_linear(input[22]),
                self.compress_linear(input[23]),
                self.compress_linear(input[24]),
                self.compress_linear(input[25]),
                self.compress_linear(input[26]),
                self.compress_linear(input[27]),
                self.compress_linear(input[28]),
                self.compress_linear(input[29]),
                self.compress_linear(input[30]),
                self.compress_linear(input[31]),
                self.compress_linear(input[32]),
                self.compress_linear(input[33]),
                self.compress_linear(input[34]),
                self.compress_linear(input[35]),
                self.compress_linear(input[36]),
                self.compress_linear(input[37]),
                self.compress_linear(input[38]),
                self.compress_linear(input[39]),
            ), dim=0)

    def compress_flexible_sequence(self, input):
        stacks = []
        for i in range(input.shape[0]):
            stacks.append(self.compress_linear(input[i]))

        return torch.stack(stacks, dim=0)

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
