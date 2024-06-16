import re
import random
import csv
import math
import sqlite3

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import Dataset

from load_spacy import load_spacy, get_vector_zero_index
from sentence_embedding_model_v3 import SentenceEmbeddingV3
from sentence_embedding_model_v7 import SentenceEmbeddingV7, CFG_V7
from dataset_utils import dataset_map


def evaluate_fn(ModelClass, config, nlp, dataloader, num_batches=1):
    def evaluate(checkpoint):
        device = config['device']
        model = ModelClass(config=config,
                              nlp=nlp,
                              inferring=True).to(device)
        model.load_state_dict(torch.load(checkpoint))
        # print(model)

        word_embedding = model.get_word_embedding()

        perpendicular_error = 0
        perpendicular_total = 0
        proportional_error = 0
        proportional_total = 0

        for batch, data in enumerate(dataloader):
            if batch >= num_batches:
                break

            sample1 = data['sample1'].to(device)
            sample2 = data['sample2'].to(device)
            r1 = data['sim_lower_r1'].to(device)
            r2 = data['sim_upper_r2'].to(device)

            sample1 = word_embedding(sample1)
            sample2 = word_embedding(sample2)

            transposed_s1 = sample1.transpose(0, 1)
            transposed_s2 = sample2.transpose(0, 1)

            en1_embedding1 = model(transposed_s1)
            en1_embedding2 = model(transposed_s2)

            pred = model.cos(en1_embedding1, en1_embedding2)

            for idx, r in enumerate(r1):
                if r >= 0.889:
                    proportional_error += abs(1 - pred[idx].item())
                    proportional_total += 1
                else:
                    perpendicular_error += abs(pred[idx].item())
                    perpendicular_total += 1

        if perpendicular_total > 0:
            error1 = perpendicular_error / perpendicular_total
            print(f'[EVALUATE] perpendicular error: {error1:0.5f} k_sample: {perpendicular_total}')

        if proportional_total > 0:
            error2 = proportional_error / proportional_total
            print(f'[EVALUATE] proportional error: {error2:0.5f} k_sample: {proportional_total}')

    return evaluate


def create_evaluating_dataloader(dataset, device, batch_size):
    conn = sqlite3.connect(f'sentence_embedding_training_data/{dataset}.db')
    ds = Dataset.from_sql( "SELECT json_data FROM record", con=conn)
    ds = ds.map(dataset_map, keep_in_memory=True, batched=True, num_proc=1)
    ds = ds.with_format('torch', device=device)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


if __name__ == '__main__':
    DEVICE = 'cuda'
    BATCH_SIZE = 512
    CFG = CFG_V7
    CFG['batch_size'] = BATCH_SIZE
    CFG['device'] = DEVICE

    nlp = load_spacy()

    # checkpoint = 'tmp/checkpoints/v3/epoch69_encoder1'
    # checkpoint = 'tmp/checkpoints/v10/epoch5_encoder1'
    checkpoint = 'tmp/checkpoints/v10/epoch10_encoder1'
    # checkpoint = 'tmp/finetuned/iterations/v3_epoch69_iter0'
    # checkpoint = 'tmp/checkpoints/batches/v9/epoch1_batch1800_encoder1'
    
    print('[INFO] checkpoint: ', checkpoint)


    dataset = 'processed-quora-duplicated-questions-train.csv'
    print(f'[INFO] dataset: {dataset}')

    dataloader = create_evaluating_dataloader(dataset, DEVICE, BATCH_SIZE)

    evaluate = evaluate_fn(SentenceEmbeddingV7, CFG, nlp, 
                           dataloader, num_batches=3)

    evaluate(checkpoint)
