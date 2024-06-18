import random
import math
import datetime
import sqlite3

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
import spacy

from sentence_embedding_model_v7 import SentenceEmbeddingV7
from train_sentence_embedding import dataset_map, CFG
from load_spacy import load_spacy


def get_database_uri(dataset_name, checkpoint_num, current_epoch, iteration):
    db_uri = f'sentence_embedding_training_data/\
{dataset_name}_v{checkpoint_num}_epoch{current_epoch}_iter{iteration - 1}.db'

    return db_uri


def get_checkpoint_filepath(checkpoint_num, current_epoch, iteration):
    checkpoint = f'tmp/finetuned/iterations/\
v{checkpoint_num}_epoch{current_epoch}_iter{iteration - 1}'

    return checkpoint


def get_new_checkpoint_filepath(checkpoint_num, current_epoch, iteration):
    return get_checkpoint_filepath(checkpoint_num, current_epoch, iteration + 1)

def train(dataloader, nlp, encoder, loss_fn, optimizer,
          config, dataset_size, epoch, checkpoint_num):
    encoder.train()

    device = config['device']
    batch_size = config['batch_size']

    y0 = torch.zeros(batch_size, dtype=torch.float32).to(device)

    word_embedding = encoder.get_word_embedding()

    for batch, data in enumerate(dataloader):

        current = (batch + 1) * BATCH_SIZE
        remains = dataset_size - current

        sample1 = data['sample1'].to(device)
        sample2 = data['sample2'].to(device)
        r1 = data['sim_lower_r1'].to(device)
        r2 = data['sim_upper_r2'].to(device)

        sample1 = word_embedding(sample1)
        sample2 = word_embedding(sample2)

        batch_size = sample1.shape[0]

        # if batch_size is not equal to BATCH_SIZE, generate_noise would crash
        if batch_size != BATCH_SIZE:
            print(f'[INFO] skip batch that is not full, \
batch_size: {batch_size}, BATCH_SIZE: {BATCH_SIZE}')
            continue

        transposed_s1 = sample1.transpose(0, 1)
        transposed_s2 = sample2.transpose(0, 1)

        embedding1 = encoder(transposed_s1)
        embedding2 = encoder(transposed_s2)

        pred2 = encoder.cos(embedding1, embedding2)

        masking_pred21 = pred2 < r1
        masking_pred21 = masking_pred21.int()
        pred21 = r1 - pred2
        pred21 = pred21.mul(masking_pred21)

        masking_pred22 = r2 < pred2
        masking_pred22 = masking_pred22.int()
        pred22 = pred2 - r2
        pred22 = pred22.mul(masking_pred22)

        pred = pred21 + pred22
        loss = loss_fn(pred, y0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch + 1) % 20 == 0 or remains < BATCH_SIZE:
            _loss = loss.item()
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'{checkpoint_num}. loss:{_loss:0.5f} \
{batch}/{current}/{dataset_size} {ts}')


BATCH_SIZE = 3096
EPOCHS = 300
CURRENT_EPOCH = 69
CHECKPOINT_NUM = 3
DATASET_SIZE = 623025
NUM_WORKERS = 7
ITERATION = 1

# BATCH_SIZE = 3
# EPOCHS = 1
# CURRENT_EPOCH = 0
# DATASET_SIZE = 11780
# NUM_WORKERS = 1

LEARNING_RATE = 0.001
DEVICE = 'cuda'
CFG['device'] = DEVICE


assert ITERATION is not None and ITERATION > 0

if __name__ == '__main__':
    print('training configurations: ', CFG)

    nlp = load_spacy()

    dataset_name = 'processed-quora-duplicated-questions-train.csv'
    db_uri = get_database_uri(dataset_name, CHECKPOINT_NUM, CURRENT_EPOCH, ITERATION)

    conn = sqlite3.connect(db_uri)
    ds = Dataset.from_sql( "SELECT json_data FROM record", con=conn)

    print(f'[INFO] Load dataset from: {db_uri}')

    encoder = SentenceEmbeddingV7(config=CFG, nlp=nlp, finetuning=True).to(DEVICE)

    print(f"[INFO] training version: {CHECKPOINT_NUM}")
    print(encoder)

    if ITERATION == 1:
        checkpoint = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder1'
    else:
        checkpoint = get_checkpoint_filepath(CHECKPOINT_NUM, CURRENT_EPOCH, ITERATION)

    print(f'[INFO] Load checkpoint: {checkpoint}')
    encoder.load_state_dict(torch.load(checkpoint))

    ds = ds.map(dataset_map, keep_in_memory=True, batched=True, num_proc=NUM_WORKERS)
    ds = ds.with_format('torch', device=DEVICE)
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # ds = ds.to_iterable_dataset(num_shards=NUM_WORKERS).map(
    #     dataset_map, batched=True)
    # ds = ds.with_format('torch')

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        print(f'\n\nEpoch {epoch}\n----------------------------------')

        # ds = ds.shuffle(
        #     seed=random.randint(1, 999),
        #     buffer_size=math.ceil(BATCH_SIZE * 4.3))
        # dataloader = DataLoader(ds, batch_size=BATCH_SIZE,
        #                         num_workers=NUM_WORKERS,
        #                         persistent_workers=True, pin_memory=True,
        #                         pin_memory_device=DEVICE)

        train(dataloader, nlp, encoder, loss_fn,
              optimizer, CFG, DATASET_SIZE, epoch, CHECKPOINT_NUM)

        encoder.eval()
        new_checkpoint = get_new_checkpoint_filepath(CHECKPOINT_NUM, CURRENT_EPOCH, ITERATION)
        torch.save( encoder.state_dict(), new_checkpoint)
