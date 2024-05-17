import math
import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import spacy

from sentence_embedding_model import SentenceEmbedding


def vectorise_sequence(nlp, fixed_sequence_length, samples):
    data = []
    vector_zero_key = nlp.vocab.strings["vector_zero"]
    vector_zero_idx = nlp.vocab.vectors.find(key=vector_zero_key)
    for sample_ in samples:
        sample = [nlp.vocab.vectors.find(key=token.lex.orth) for token in nlp(sample_)]
        length = len(sample)
        padlen = fixed_sequence_length - length
        if padlen >= 0:
            padding = [vector_zero_idx for _ in range(padlen)]
            sample = sample + padding
        else:
            sample = sample[0:fixed_sequence_length]

        data.append(sample)

    return data


def dataset_map_fn(nlp, fixed_sequence_length):

    def dataset_map(samples):
        sample1 = vectorise_sequence(nlp, fixed_sequence_length, samples['sample1'])
        sample2 = vectorise_sequence(nlp, fixed_sequence_length, samples['sample2'])

        return {
            'sample1': sample1,
            'sample2': sample2,
            'random_similarity': samples['random_similarity']
        }

    return dataset_map


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


def train(dataloader, word_embedding, encoder1, encoder2, loss_fn, optimizer1, optimizer2,
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
    y1 = torch.ones(batch_size, dtype=torch.float32).to(device)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for batch, data in enumerate(dataloader):
        sample1 = data['sample1'].to(device)
        sample1 = word_embedding(sample1)

        sample2 = data['sample2'].to(device)
        sample2 = word_embedding(sample2)

        random_similarity = data['random_similarity'].to(device)

        batch_size = sample1.shape[0]

        # if batch_size is not equal to BATCH_SIZE, generate_noise would crash
        if batch_size != BATCH_SIZE:
            print(f'[INFO] skip batch that is not full,\
 batch_size: {batch_size}')
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

        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()

        if checkpoint_num == 1:
            metric_version = 1

            en1_embedding1 = encoder1(noised_transposed_s1)
            en2_embedding1 = encoder2(noised_transposed_s1)
            pred2 = cos(en1_embedding1, en2_embedding1)
            loss2 = loss_fn(pred2, y1)

            en1_embedding1 = encoder1(noised_transposed_s1)
            en1_embedding2 = encoder1(noised_transposed_s2)
            pred4 = cos(en1_embedding1, en1_embedding2)
            loss4 = loss_fn(pred4, random_similarity)

            loss24 = loss2.add(loss4)
            loss24.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()

            en1_embedding2 = encoder1(noised_transposed_s2)
            en2_embedding2 = encoder2(noised_transposed_s2)
            pred3 = cos(en2_embedding2, en1_embedding2)
            loss3 = loss_fn(pred3, y1)

            en2_embedding1 = encoder2(noised_transposed_s1)
            en2_embedding2 = encoder2(noised_transposed_s2)
            pred5 = cos(en2_embedding1, en2_embedding2)
            loss5 = loss_fn(pred5, random_similarity)

            loss35 = loss3.add(loss5)
            loss35.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()

        if checkpoint_num == 2:
            metric_version = 2

            en1_embedding1 = encoder1(noised_transposed_s1)
            en2_embedding1 = encoder2(noised_transposed_s1)
            pred2 = cos(en1_embedding1, en2_embedding1)
            loss2 = loss_fn(pred2, y1)

            en1_embedding1 = encoder1(noised_transposed_s1)
            en1_embedding2 = encoder1(noised_transposed_s2)
            pred4 = cos(en1_embedding1, en1_embedding2)
            loss4 = loss_fn(pred4, random_similarity)

            en1_embedding2 = encoder1(noised_transposed_s2)
            en2_embedding2 = encoder2(noised_transposed_s2)
            pred3 = cos(en2_embedding2, en1_embedding2)
            loss3 = loss_fn(pred3, y1)

            en2_embedding1 = encoder2(noised_transposed_s1)
            en2_embedding2 = encoder2(noised_transposed_s2)
            pred5 = cos(en2_embedding1, en2_embedding2)
            loss5 = loss_fn(pred5, random_similarity)

            loss = loss2.add(loss4).add(loss3).add(loss5)
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()

        
        else:
            metric_version = -1

        ### Extra foward
        if batch % 3 == 0:
            en2_embedding1 = encoder2(noised_transposed_s1)
            en2_embedding2 = encoder2(noised_transposed_s2)
            pred5 = cos(en2_embedding1, en2_embedding2)
            loss5 = loss_fn(pred5, random_similarity)

            loss5.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        current = (batch + 1) * BATCH_SIZE
        remains = dataset_size - current

        if batch % 100 == 0 or remains < BATCH_SIZE:
            loss1, loss2, loss3, loss4, loss5  = (
                    loss1.item(), loss2.item(), loss3.item(), loss4.item(),
                    loss5.item())

            loss24 = loss2 + loss4
            loss35 = loss3 + loss5
            total_loss = loss1 + loss24 + loss35

            ts = datetime.datetime.now().strftime('%H:%M:%S')

            print(f'{metric_version}. loss1:{loss1:0.3f}  loss2:{loss2:0.3f} loss3:{loss3:0.3f} \
loss4:{loss4:0.3f} loss5:{loss5:0.3f}  loss24:{loss24:0.3f} loss35:{loss35:0.3f} \
total:{total_loss:0.3f}  {batch}/{current}/{dataset_size} {ts}')

            # torch.save(encoder1.state_dict(),
            #     f'tmp/checkpoints/batches/epoch{epoch}_batch{batch}_encoder1')
            # torch.save(encoder2.state_dict(),
            #     f'tmp/checkpoints/batches/epoch{epoch}_batch{batch}_encoder2')


if __name__ == '__main__':
    FIXED_SEQUENCE_LENGTH = 40
    BATCH_SIZE = 1024
    EPOCHS = 200
    CURRENT_EPOCH = 0
    # BATCH_SIZE = 3
    # EPOCHS = 3
    # CURRENT_EPOCH = 0
    DATASET_SIZE = 1466076

    LEARNING_RATE = 0.005
    DEVICE = 'cuda'
    NUM_WORKERS = 8
    CHECKPOINT_NUM = 2

    CFG = {
        'embed_size': 300,
        'hidden_size1': 32,
        'hidden_size2': 128,
        'dropout1': 0.11,
        'dropout2': 0.83,
        'num_layers1': 2,
        'num_layers2': 4,
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'fixed_sequence_length': FIXED_SEQUENCE_LENGTH,
        'token_noise_magnitue': 0.47,
        'sequence_noise_ratio': 0.41
    }
    print('training configurations: ', CFG)

    nlp = spacy.load('en_core_web_lg')
    nlp.vocab.vectors.resize((nlp.vocab.vectors.shape[0] + 1, nlp.vocab.vectors.shape[1]))
    item_id = nlp.vocab.strings.add("vector_zero")
    nlp.vocab.vectors.add(item_id, vector=np.zeros(300, dtype=np.float32))

    dataset_map = dataset_map_fn(nlp, FIXED_SEQUENCE_LENGTH)

    data_files = [
        # 'sentence_embedding_training_data/samples.csv',
        'sentence_embedding_training_data/abcnews-date-text.csv',
        'sentence_embedding_training_data/cnbc_headlines.csv',
        'sentence_embedding_training_data/guardian_headlines.csv',
        'sentence_embedding_training_data/reuters_headlines.csv',
        'sentence_embedding_training_data/processed-imdb-movie-rating.csv',
    ]
    ds = load_dataset(
        'csv',
        data_files=data_files,
        delimiter='\t', split='train'
    )

    # ds = ds.to_iterable_dataset(num_shards=NUM_WORKERS).map(
    #     dataset_map, batched=True).shuffle(
    #         seed=42, buffer_size=math.ceil(BATCH_SIZE * 2.1))
    # ds = ds.with_format('torch')
    # dataloader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    #                         persistent_workers=True, pin_memory=True,
    #                         pin_memory_device=DEVICE)

    ds = ds.map(dataset_map, keep_in_memory=True, batched=True, num_proc=NUM_WORKERS)
    ds = ds.with_format('torch', device=DEVICE)
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    word_embedding = nn.Embedding.from_pretrained(get_word_vector_matrix(nlp, DEVICE))

    encoder1 = SentenceEmbedding(CFG).to(DEVICE)
    encoder2 = SentenceEmbedding(CFG).to(DEVICE)

    if CURRENT_EPOCH > 0:
        checkpoint1 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder1'
        checkpoint2 = f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{CURRENT_EPOCH}_encoder2'
        print(f'Load checkpoint: {checkpoint1}')
        print(f'Load checkpoint: {checkpoint2}')
        encoder1.load_state_dict(torch.load(checkpoint1))
        encoder2.load_state_dict(torch.load(checkpoint2))

    loss_fn = nn.L1Loss()
    optimizer1 = torch.optim.Adam(encoder1.parameters(), lr=LEARNING_RATE)
    optimizer2 = torch.optim.Adam(encoder2.parameters(), lr=LEARNING_RATE)

    for epoch in range(CURRENT_EPOCH + 1, EPOCHS + CURRENT_EPOCH + 1):
        print(f'\n\nEpoch {epoch}\n----------------------------------')

        train(dataloader, word_embedding, encoder1, encoder2, loss_fn,
              optimizer1, optimizer2, CFG, DATASET_SIZE, epoch, CHECKPOINT_NUM)

        torch.save(encoder1.state_dict(),
                   f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{epoch}_encoder1')
        torch.save(encoder2.state_dict(),
                   f'tmp/checkpoints/v{CHECKPOINT_NUM}/epoch{epoch}_encoder2')

        torch.save(encoder1.state_dict(), f'tmp/encoder1_v{CHECKPOINT_NUM}')
        torch.save(encoder2.state_dict(), f'tmp/encoder2_v{CHECKPOINT_NUM}')
