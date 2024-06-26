import random
import math
import datetime
import sqlite3

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value
import numpy as np
import spacy

from sentence_embedding_model_v7 import SentenceEmbeddingV7, CFG_V7
from load_spacy import load_spacy, get_vector_zero_index, translate_sequence
from evaluation import evaluate_fn, create_evaluating_dataloader
from dataset_utils import dataset_map


def apply_noise(sample, token_noise_magnitue, threshold, device):
    fixed_sequence_length, batch_size, embed_size = sample.shape

    sample_length_filter = sample != torch.zeros(
        sample.shape, dtype=torch.float32).to(device)

    noise_masking = torch.rand(
        (fixed_sequence_length, batch_size, 1), dtype=torch.float32).to(device)
    noise_masking = noise_masking < threshold

    noise = torch.rand(sample.shape, dtype=torch.float32).to(device)
    noise = noise.mul(token_noise_magnitue).mul(sample)
    noise = noise.mul(noise_masking.int()).mul(sample_length_filter.int())

    return sample.add(noise)


def train(dataloader, nlp, encoder1, encoder2, loss_fn, optimizer1, optimizer2,
          config, dataset_size, epoch, checkpoint_num):
    encoder1.train()
    encoder2.train()

    device = config['device']
    batch_size = config['batch_size']
    fixed_sequence_length = config['fixed_sequence_length']

    token_noise_magnitue = torch.tensor(
        config['token_noise_magnitue'], dtype=torch.float32).to(device)
    sequence_noise_ratio = torch.tensor(
        config['sequence_noise_ratio'], dtype=torch.float32).to(device)

    y0 = torch.zeros(batch_size, dtype=torch.float32).to(device)

    word_embedding = encoder1.get_word_embedding()
    idx = 1

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

        # if batch_size is not equal to BATCH_SIZE, apply_noise would crash
        if batch_size != BATCH_SIZE:
            print(f'[INFO] skip batch that is not full, \
batch_size: {batch_size}, BATCH_SIZE: {BATCH_SIZE}')
            continue

        threshold = torch.zeros((fixed_sequence_length, batch_size), dtype=torch.float32).to(device)
        threshold = threshold.add(PROPOTIONAL_THRESHOLD).sub(1e-3)
        noise_preferring = r1 > threshold
        noise_preferring = noise_preferring.int()
        noise_preferring = noise_preferring.unsqueeze(dim=2)

        transposed_s1 = sample1.transpose(0, 1)
        transposed_s2 = sample2.transpose(0, 1)

        if random.randint(0, 1) == 0:
            en1_embedding1 = encoder1(transposed_s1, noise_preferring=noise_preferring)
            en1_embedding2 = encoder1(transposed_s2, noise_preferring=noise_preferring)

            en2_embedding1 = encoder2(transposed_s1, noise_preferring=1)
            en2_embedding2 = encoder2(transposed_s2, noise_preferring=1)

            pred1 = encoder1.cos(en2_embedding1, en2_embedding2)
            pred2 = encoder1.cos(en1_embedding1, en1_embedding2)
        else:
            en1_embedding1 = encoder1(transposed_s1, noise_preferring=1)
            en1_embedding2 = encoder1(transposed_s2, noise_preferring=1)

            en2_embedding1 = encoder2(transposed_s1, noise_preferring=noise_preferring)
            en2_embedding2 = encoder2(transposed_s2, noise_preferring=noise_preferring)

            pred1 = encoder1.cos(en1_embedding1, en1_embedding2)
            pred2 = encoder1.cos(en2_embedding1, en2_embedding2)

        diff1 = pred1.sub(pred2).abs()
        loss1 = loss_fn(diff1, y0)

        masking_pred21 = pred2 < r1
        masking_pred21 = masking_pred21.int()
        pred21 = r1 - pred2
        pred21 = pred21.mul(masking_pred21)

        masking_pred22 = r2 < pred2
        masking_pred22 = masking_pred22.int()
        pred22 = pred2 - r2
        pred22 = pred22.mul(masking_pred22)

        pred2 = pred21 + pred22
        pred2 = pred2.mul(3.3)
        loss2 = loss_fn(pred2, y0)

        loss = loss1.add(loss2)

        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()

        if (batch + 1) % 100 == 0 or remains < BATCH_SIZE:
            _loss1, _loss2, total_loss  = (
                    loss1.item(), loss2.item(), loss.item())

            ts = datetime.datetime.now().strftime('%H:%M:%S')

            print(f'{checkpoint_num}. loss1:{_loss1:0.5f}  loss2:{_loss2:0.5f} \
total:{total_loss:0.5f}  {batch}/{current}/{dataset_size} {ts}')

        # if (batch + 1) % 10000 == 0 or remains < BATCH_SIZE:
            torch.save(encoder1.state_dict(),
                    f'tmp/checkpoints/batches/v{checkpoint_num}/epoch{epoch}_batch{batch + 1}_encoder1')
            torch.save(encoder2.state_dict(),
                    f'tmp/checkpoints/batches/v{checkpoint_num}/epoch{epoch}_batch{batch + 1}_encoder2')


SIM_LOWER_R1 = -0.375
SIM_UPPER_R2 = 0.425
PROPOTIONAL_THRESHOLD = 0.89
IDENTICAL_THRESHOLD = 0.95

FIXED_SEQUENCE_LENGTH = 40

SWITCH = 1
BATCH_SIZE = 3500
EPOCHS = 3
CURRENT_EPOCH = 5
DATASET_SIZE = 104434491
NUM_WORKERS = 6
CHECKPOINT_NUM = 19

# SWITCH = 0
# BATCH_SIZE = 64
# EPOCHS = 3
# CURRENT_EPOCH = 0
# DATASET_SIZE = 24738
# NUM_WORKERS = 1
# CHECKPOINT_NUM = 99

LEARNING_RATE = 0.0002
DEVICE = 'cuda'


CFG = CFG_V7
CFG['device'] = DEVICE
CFG['batch_size'] = BATCH_SIZE
CFG['fixed_sequence_length'] = FIXED_SEQUENCE_LENGTH

if __name__ == '__main__':
    print('training configurations: ', CFG)
    switch = SWITCH

    nlp = load_spacy()

    if switch != 0:
        db_uri = '/archives/sentence_embedding_training_data/sqlite_file.db'
        # db_uri = '/archives/sentence_embedding_training_data/urlsf_subset00.db'
    else:
        db_uri = 'sentence_embedding_training_data/guardian_headlines.txt.db'

    conn = sqlite3.connect(db_uri)
    ds = Dataset.from_sql( "SELECT json_data FROM record", con=conn)

    encoder1 = SentenceEmbeddingV7(config=CFG, nlp=nlp).to(DEVICE)
    encoder2 = SentenceEmbeddingV7(config=CFG, nlp=nlp, dropout=CFG['asym_dropout']).to(DEVICE)
    print(encoder1)

    print(f"[INFO] training version: {CHECKPOINT_NUM}")
    print(f"[INFO] encoder1's dropout: {encoder1.dropout_ratio}")
    print(f"[INFO] encoder2's dropout: {encoder2.dropout_ratio}")
    print(f"[INFO] encoder's compress_size1: {encoder1.compress_size1}")
    print(f"[INFO] encoder's hidden_size1: {encoder1.hidden_size1}")
    print(f"[INFO] encoder's hidden_size2: {encoder1.hidden_size2}")
    print(f"[INFO] encoder's num_layers1: {encoder1.num_layers1}")
    print(f"[INFO] encoder's num_layers2: {encoder1.num_layers2}")
    print(f"[INFO] batch size: {BATCH_SIZE}")
    print(f"[INFO] learning rate: {LEARNING_RATE}")

    if CURRENT_EPOCH > 0:
        checkpoint1 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder1'
        checkpoint2 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder2'
        print(f'Load checkpoint: {checkpoint1}')
        print(f'Load checkpoint: {checkpoint2}')
        encoder1.load_state_dict(torch.load(checkpoint1))
        encoder2.load_state_dict(torch.load(checkpoint2))

    print(f'[INFO] training data: {db_uri}')

    if switch != 0:
        ds = ds.to_iterable_dataset(num_shards=NUM_WORKERS).map(dataset_map, batched=True)    
        ds = ds.with_format('torch')
    else:
        ds = ds.map(dataset_map, keep_in_memory=True, batched=True, num_proc=NUM_WORKERS)
        ds = ds.with_format('torch', device=DEVICE)

    eval_dataset = 'processed-quora-duplicated-questions-train.csv'
    eval_dataloader = create_evaluating_dataloader(eval_dataset, DEVICE, BATCH_SIZE)
    evaluate = evaluate_fn(SentenceEmbeddingV7, CFG, nlp, 
                           eval_dataloader, num_batches=100)

    loss_fn = nn.L1Loss()
    optimizer1 = torch.optim.Adam(encoder1.parameters(), lr=LEARNING_RATE)
    optimizer2 = torch.optim.Adam(encoder2.parameters(), lr=LEARNING_RATE)

    for epoch in range(CURRENT_EPOCH + 1, EPOCHS + CURRENT_EPOCH + 1):
        print(f'\n\nEpoch {epoch}\n----------------------------------')


        if switch != 0:
            ds = ds.shuffle(seed=random.randint(1, 999), buffer_size=math.ceil(BATCH_SIZE * 20))
            dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                    prefetch_factor=NUM_WORKERS*20, persistent_workers=True,
                                    pin_memory=True, pin_memory_device=DEVICE, drop_last=True)
        else:
            dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        train(dataloader, nlp, encoder1, encoder2, loss_fn,
              optimizer1, optimizer2, CFG, DATASET_SIZE, epoch, CHECKPOINT_NUM)

        checkpoint1 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{epoch}_encoder1'
        encoder1.eval()
        torch.save(encoder1.state_dict(), checkpoint1)

        checkpoint2 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{epoch}_encoder2'
        encoder2.eval()
        torch.save(encoder2.state_dict(), checkpoint2)

        print('\n')
        evaluate(checkpoint1)
        print('\n')
        evaluate(checkpoint2)
