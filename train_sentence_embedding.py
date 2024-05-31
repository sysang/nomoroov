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
import msgspec

from sentence_embedding_model import SentenceEmbedding
from data_schema import SentencePair


def dataset_map(samples):
    sample1 = []
    sample2 = []
    r1 = []
    r2 = []

    for sample in samples['json_data']:
        pair = msgspec.json.decode(sample, type=SentencePair)
        sample1.append(pair.sample1)
        sample2.append(pair.sample2)
        r1.append(pair.sim_lower_r1)
        r2.append(pair.sim_upper_r2)

    return {
        'sample1': sample1,
        'sample2': sample2,
        'sim_lower_r1': r1,
        'sim_upper_r2': r2,
    }


def generate_noise(sample, token_noise_magnitue, threshold, device):
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


def get_word_vector_matrix(nlp, device):
    wv = map(lambda item: item[1], nlp.vocab.vectors.items())
    wv = np.array(list(wv))
    return torch.FloatTensor(np.array(wv, dtype=np.float32)).to(device)


def train(dataloader, word_embedding, nlp, encoder1, encoder2, loss_fn, optimizer1, optimizer2,
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

    threshold = torch.ones(
        (fixed_sequence_length, batch_size, 1),
        dtype=torch.float32).to(device).mul(sequence_noise_ratio)

    y0 = torch.zeros(batch_size, dtype=torch.float32).to(device)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for batch, data in enumerate(dataloader):

        current = (batch + 1) * BATCH_SIZE
        remains = dataset_size - current

        sample1 = data['sample1'].to(device)
        sample2 = data['sample2'].to(device)
        r1 = data['sim_lower_r1'].to(device)
        r2 = data['sim_upper_r2'].to(device)

        # print('sample1[0]: ', sample1[0])
        # print('sample2[0]: ', sample2[0])
        # print('r1[0]: ', r1[0])
        # print('r2[0]: ', r2[0])

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

        noised_transposed_s1 = generate_noise(
            transposed_s1, token_noise_magnitue, threshold, device)
        noised_transposed_s2 = generate_noise(
            transposed_s2, token_noise_magnitue, threshold, device)

        en1_embedding1 = encoder1(noised_transposed_s1)
        en1_embedding2 = encoder1(noised_transposed_s2)
        en2_embedding1 = encoder2(noised_transposed_s1)
        en2_embedding2 = encoder2(noised_transposed_s2)

        cosim1 = cos(en1_embedding1, en1_embedding2)
        cosim2 = cos(en2_embedding1, en2_embedding2)
        pred1 = cosim1.sub(cosim2).abs()
        loss1 = loss_fn(pred1, y0)

        if random.randint(0, 1) == 0:
            pred2 = cos(en1_embedding1, en1_embedding2)
        else:
            pred2 = cos(en2_embedding1, en2_embedding2)

        masking_pred21 = pred2 < r1
        masking_pred21 = masking_pred21.int()
        pred21 = r1 - pred2
        pred21 = pred21.mul(masking_pred21)

        masking_pred22 = r2 < pred2
        masking_pred22 = masking_pred22.int()
        pred22 = pred2 - r2
        pred22 = pred22.mul(masking_pred22)

        pred2 = pred21 + pred22
        loss2 = loss_fn(pred2, y0)

        loss = loss1.add(loss2)

        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        
        if (batch + 1) % 200 == 0 or remains < BATCH_SIZE:
            _loss1, _loss2, total_loss  = (
                    loss1.item(), loss2.item(), loss.item())

            ts = datetime.datetime.now().strftime('%H:%M:%S')

            print(f'{checkpoint_num}. loss1:{_loss1:0.5f}  loss2:{_loss2:0.5f} \
total:{total_loss:0.5f}  {batch}/{current}/{dataset_size} {ts}')

            torch.save(encoder1.state_dict(),
                    f'tmp/checkpoints/batches/v{checkpoint_num}/epoch{epoch}_batch{batch + 1}_encoder1')
            torch.save(encoder2.state_dict(),
                    f'tmp/checkpoints/batches/v{checkpoint_num}/epoch{epoch}_batch{batch + 1}_encoder2')


FIXED_SEQUENCE_LENGTH = 40

BATCH_SIZE = 2048
EPOCHS = 10
CURRENT_EPOCH = 74
DATASET_SIZE = 20717560
NUM_WORKERS = 8

# BATCH_SIZE = 3
# EPOCHS = 1
# CURRENT_EPOCH = 0
# DATASET_SIZE = 11780
# NUM_WORKERS = 1

LEARNING_RATE = 0.001
DEVICE = 'cuda'

CHECKPOINT_NUM = 3
ASYM_DROPOUT1 = 0.21
ASYM_DROPOUT2 = 0.93

CFG = {
    'embed_size': 300,
    # 'hidden_size1': 32,   # v1
    # 'hidden_size2': 128,  # v1
    'hidden_size1': 16,   # v2
    'hidden_size2': 64,  # v2
    'dropout1': 0.17,
    'dropout2': 0.81,
    'num_layers1': 2,
    # 'num_layers2': 2,     # v1
    'num_layers2': 3,     # v2
    'device': DEVICE,
    'batch_size': BATCH_SIZE,
    'fixed_sequence_length': FIXED_SEQUENCE_LENGTH,
    'token_noise_magnitue': 4.9,
    'sequence_noise_ratio': 0.67
}

if __name__ == '__main__':
    print('training configurations: ', CFG)

    nlp = spacy.load('en_core_web_lg')
    nlp.vocab.vectors.resize((nlp.vocab.vectors.shape[0] + 1, nlp.vocab.vectors.shape[1]))
    item_id = nlp.vocab.strings.add("vector_zero")
    nlp.vocab.vectors.add(item_id, vector=np.zeros(300, dtype=np.float32))

    word_embedding = nn.Embedding.from_pretrained(get_word_vector_matrix(nlp, DEVICE))

    # conn = sqlite3.connect('sentence_embedding_training_data/guardian_headlines.txt.db')
    conn = sqlite3.connect('sentence_embedding_training_data/sqlite_file.db')
    ds = Dataset.from_sql( "SELECT json_data FROM record", con=conn)

    encoder1 = SentenceEmbedding(CFG).to(DEVICE)
    encoder2 = SentenceEmbedding(CFG, dropout1=ASYM_DROPOUT1, dropout2=ASYM_DROPOUT2).to(DEVICE)

    print(f"[INFO] training version: {CHECKPOINT_NUM}")
    print(f"[INFO] encoder1's dropout1: {encoder1.dropout1}")
    print(f"[INFO] encoder1's dropout2: {encoder1.dropout2}")
    print(f"[INFO] encoder2's dropout1: {encoder2.dropout1}")
    print(f"[INFO] encoder2's dropout2: {encoder2.dropout2}")
    print(f"[INFO] encoder's hidden_size1: {encoder1.hidden_size1}")
    print(f"[INFO] encoder's hidden_size2: {encoder1.hidden_size2}")
    print(f"[INFO] encoder's num_layers1: {encoder1.num_layers1}")
    print(f"[INFO] encoder's num_layers2: {encoder1.num_layers2}")

    if CURRENT_EPOCH > 0:
        checkpoint1 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder1'
        checkpoint2 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder2'
        print(f'Load checkpoint: {checkpoint1}')
        print(f'Load checkpoint: {checkpoint2}')
        encoder1.load_state_dict(torch.load(checkpoint1))
        encoder2.load_state_dict(torch.load(checkpoint2))

    ds = ds.to_iterable_dataset(num_shards=NUM_WORKERS).map(dataset_map, batched=True)    
    ds = ds.with_format('torch')
    # ds = ds.map(dataset_map, keep_in_memory=True, batched=True, num_proc=NUM_WORKERS)
    # ds = ds.with_format('torch', device=DEVICE)
    # dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.L1Loss()
    optimizer1 = torch.optim.Adam(encoder1.parameters(), lr=LEARNING_RATE)
    optimizer2 = torch.optim.Adam(encoder2.parameters(), lr=LEARNING_RATE)

    for epoch in range(CURRENT_EPOCH + 1, EPOCHS + CURRENT_EPOCH + 1):
        print(f'\n\nEpoch {epoch}\n----------------------------------')


        ds = ds.shuffle(seed=random.randint(1, 999), buffer_size=math.ceil(BATCH_SIZE * 4.3))
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                persistent_workers=True, pin_memory=True,
                                pin_memory_device=DEVICE)

        train(dataloader, word_embedding, nlp, encoder1, encoder2, loss_fn,
              optimizer1, optimizer2, CFG, DATASET_SIZE, epoch, CHECKPOINT_NUM)

        torch.save(encoder1.state_dict(),
                   f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{epoch}_encoder1')
        torch.save(encoder2.state_dict(),
                   f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{epoch}_encoder2')

        torch.save(encoder1.state_dict(), f'tmp/encoder1_v{CHECKPOINT_NUM}')
        torch.save(encoder2.state_dict(), f'tmp/encoder2_v{CHECKPOINT_NUM}')
