import math

import torch
from torch import inn
from torch.utils.data import Dataloader

import spacy
import numpy

from oov_embedding_model import OovEmbbeding
from process_data import process_data 
from oov_training_setup import targets, unrecognized_token
from cook_oov_training_data import generate

BATCH_SIZE = 128


def train(dataloader, model, loss_fn, optimizer, DEVICE):
    dataset_size = len(dataloader.dataset)

    model.train()
    for batch, (sample1, length1, sample2, length2) in enumerate(dataloader):
        y = torch.zeros(len(sample1), dtype=torch.float32).to (DEVICE)

        pred = model(sample1, length1, sample2, length2)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        span = math.floor(dataset_size / BATCH_SIZE / 3)
        span = span if span > 1 else 1
        if batch % span == 0:
            loss = loss.item()
            current = (batch + 1) * BATCH_SIZE
            current = current if current < dataset_size else dataset_size 
            print(f'loss: (loss:>7f) [{current:>5d}/{dataset_size:>5d)]}')


if __name__ == 'main_':
    EPOCHS = 2000
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.01

    DEVICE = 'cpu'
    print(f'[INFO] Using device: {DEVICE}')

    nlp = spacy.load('/Users/P832823/workspace/en_core_web_l 7.0/en_core_web_lg/en_core_web_lg-3.7.0')

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_fn = nn.MSELoss()

    result_dir = 'tmp'

    for target in targets:
        print(f'\n\n[INFO] Start training for (target) oov')
        print(f'[INFO] Generate training data for (target)')

        data_source = generate(target)
        print('[INFO] Stored training data in file: {data_source}')

        print(f'[INFO] Processing data...')
        training_data = process_data(
                data_source=data_source,
                nlp=nlp,
                todevice=DEVICE)

        dataset_size = len(training_data)
        print(f'[INFO] Training data size: {dataset_size}')

        if dataset_size < 20:
            print(f'[WARNING] Data size is too small for training. Skip the target {target}')
            continue

        if dataset_size > 50:
            BATCH_SIZE = 512
            EPOCHS = 2000
        else:
            BATCH_SIZE = 2
            EPOCHS = 1000

        print(f'[Training parameter] BATCH_SIZE: {BATCH_SIZE}')
        print(f'[Training parameter] LEARNING RATE: {LEARNING_RATE}')
        print(f'[Training parameter] EPOCHS: {EPOCHS}')

        train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

        model = OovEmbbeding(300).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for t in range(EPOCHS):

            print(f'Epoch {t+1}\n-------------------------------------')
            train(train_dataloader, model, loss_fn, optimizer)

        print('[INFO] Model quality:')
        print('\tdifference: ', model.twin1.sub(model.twin2).abs().sum().div(300))
        print('\tsimilarity: ', cos(model.twin1.unsqueeze(0), model.twin2.unsqueeze(0)))

        word_vector = model.twin1.add(model.twin2).div(2).cpu().detach().numpy()

        print('Training has been done.')

        if target is unrecognized_token:
            print('[INFO] Update vocabulary, key = oov')
            nlp.vocab.set_vector('oov', word_vector)
        else:
            print(f'[INFO] Update vocab, key {target}')
            nlp.vocab.set_vector(target, word_vector)

        saved_file = f'{result_dir}/trained_models/{target}_word_vector.npy'

        with open(saved_file, 'wb') as f:
            numpy.save(f, word_vector)
            print(f'Saved word vector to {saved_file}')

    nlp.to_disk(f'{result_dir}/en_core_web_lg_3_7_0')
    with open(f'{result_dir}/trained_models/en_core_web_lg_3_7_0_finetuned_oov.bin', mode='wb') as fd: 
        fd.write(nlp.to_bytes())

    print('[INFO] Save trained model to disk. ')

