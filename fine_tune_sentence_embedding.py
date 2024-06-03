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

from sentence_embedding_model import SentenceEmbedding
from train_sentence_embedding import dataset_map, get_word_vector_matrix


def train(dataloader, word_embedding, nlp, encoder, loss_fn, optimizer,
          config, dataset_size, epoch, checkpoint_num):
    encoder.train()

    device = config['device']
    batch_size = config['batch_size']

    y0 = torch.zeros(batch_size, dtype=torch.float32).to(device)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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

        pred2 = cos(embedding1, embedding2)

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


FIXED_SEQUENCE_LENGTH = 40

BATCH_SIZE = 3096
EPOCHS = 300
CURRENT_EPOCH = 69
CHECKPOINT_NUM = 3
DATASET_SIZE = 623025
NUM_WORKERS = 8
ITERATION = 1

# BATCH_SIZE = 3
# EPOCHS = 1
# CURRENT_EPOCH = 0
# DATASET_SIZE = 11780
# NUM_WORKERS = 1

LEARNING_RATE = 0.001
DEVICE = 'cuda'


CFG = {
    'embed_size': 300,
    # 'hidden_size1': 32,   # v1
    # 'hidden_size2': 128,  # v1
    'hidden_size1': 16,   # v2
    'hidden_size2': 64,  # v2
    'dropout1': 0.21,
    'dropout2': 0.93,
    'num_layers1': 2,
    # 'num_layers2': 2,     # v1
    'num_layers2': 3,     # v2
    'device': DEVICE,
    'batch_size': BATCH_SIZE,
    'fixed_sequence_length': FIXED_SEQUENCE_LENGTH,
    'token_noise_magnitue': 4.9,
    'sequence_noise_ratio': 0.67
}

assert ITERATION is not None

if __name__ == '__main__':
    print('training configurations: ', CFG)

    nlp = spacy.load('en_core_web_lg')
    nlp.vocab.vectors.resize((nlp.vocab.vectors.shape[0] + 1,
                              nlp.vocab.vectors.shape[1]))
    item_id = nlp.vocab.strings.add("vector_zero")
    nlp.vocab.vectors.add(item_id, vector=np.zeros(300, dtype=np.float32))

    word_embedding = nn.Embedding.from_pretrained(
        get_word_vector_matrix(nlp, DEVICE))

    dataset_name = 'processed-quora-duplicated-questions-train.csv'
    if ITERATION == 0 :
        db_uri = f'sentence_embedding_training_data/{dataset_name}.db'
    else:
        db_uri = f'sentence_embedding_training_data/{dataset_name}_iter{ITERATION - 1}.db'

    conn = sqlite3.connect(db_uri)
    ds = Dataset.from_sql( "SELECT json_data FROM record", con=conn)

    print(f'[INFO] Load dataset from: {db_uri}')

    encoder = SentenceEmbedding(CFG, finetuning=True).to(DEVICE)

    print(f"[INFO] training version: {CHECKPOINT_NUM}")
    print(f"[INFO] encoder's dropout1: {encoder.dropout1}")
    print(f"[INFO] encoder's dropout2: {encoder.dropout2}")
    print(f"[INFO] encoder's hidden_size1: {encoder.hidden_size1}")
    print(f"[INFO] encoder's hidden_size2: {encoder.hidden_size2}")
    print(f"[INFO] encoder's num_layers1: {encoder.num_layers1}")
    print(f"[INFO] encoder's num_layers2: {encoder.num_layers2}")

    if ITERATION == 0:
        checkpoint = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder1'
    else:
        checkpoint = f'tmp/finetuned/iterations/v{CHECKPOINT_NUM}_epoch{CURRENT_EPOCH}_iter{ITERATION - 1}'

    print(f'[INFO] Load checkpoint: {checkpoint}')
    encoder.load_state_dict(torch.load(checkpoint))

    print('[INFO] Number of parameters: ', sum( p.numel() for p in encoder.parameters() if p.requires_grad))

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

        train(dataloader, word_embedding, nlp, encoder, loss_fn,
              optimizer, CFG, DATASET_SIZE, epoch, CHECKPOINT_NUM)

        torch.save(
            encoder.state_dict(),
            f'tmp/finetuned/epoches/v{CHECKPOINT_NUM}_epoch{CURRENT_EPOCH}_iter{ITERATION}_epoch{epoch}')
        torch.save(
            encoder.state_dict(),
            f'tmp/finetuned/iterations/v{CHECKPOINT_NUM}_epoch{CURRENT_EPOCH}_iter{ITERATION}')
