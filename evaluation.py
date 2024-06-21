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
        overall_error = 0

        similarity_threshold1 = 0.29
        correct1 = 0
        false_similarity1 = 0
        true_similarity1 = 0
        similarity_threshold2 = 0.51
        correct2 = 0
        false_similarity2 = 0
        true_similarity2 = 0
        similarity_threshold3 = 0.83
        correct3 = 0
        false_similarity3 = 0
        true_similarity3 = 0

        for batch, data in enumerate(dataloader):
            if num_batches != -1 and batch >= num_batches:
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
                similarity = pred[idx].item()
                if r >= 0.889:
                    proportional_error += abs(1 - similarity)
                    overall_error += abs(1 - similarity)
                    proportional_total += 1

                    correct1 += int(similarity > similarity_threshold1)
                    correct2 += int(similarity > similarity_threshold2)
                    correct3 += int(similarity > similarity_threshold3)

                    true_similarity1 += int(similarity > similarity_threshold1)
                    true_similarity2 += int(similarity > similarity_threshold2)
                    true_similarity3 += int(similarity > similarity_threshold3)
                else:
                    perpendicular_error += abs(similarity)
                    overall_error += abs(similarity)
                    perpendicular_total += 1

                    correct1 += int(similarity <= similarity_threshold1)
                    correct2 += int(similarity <= similarity_threshold2)
                    correct3 += int(similarity <= similarity_threshold3)

                    false_similarity1 += int(similarity > similarity_threshold1)
                    false_similarity2 += int(similarity > similarity_threshold2)
                    false_similarity3 += int(similarity > similarity_threshold3)

        total_samples = perpendicular_total + proportional_total

        if perpendicular_total > 0:
            error1 = perpendicular_error / perpendicular_total
            print(f'[EVALUATE] perpendicular error:\t\t{error1:0.5f}\t(over {perpendicular_total} samples)')

        if proportional_total > 0:
            error2 = proportional_error / proportional_total
            print(f'[EVALUATE] proportional error:\t\t{error2:0.5f}\t(over {proportional_total} samples)')

        print(f'[EVALUATE] perpendicular>>proportional:\t{(error1 + error2):0.5f}\t(over distribution {perpendicular_total} >> {proportional_total})')

        structural_error = (perpendicular_error + error2 * perpendicular_total) / 2 / perpendicular_total
        print(f'[EVALUATE] structural error*:\t\t{structural_error:0.5f}\t(over equal distribution {2 * perpendicular_total})')

        if perpendicular_total > 0 and proportional_total > 0:
            error3 = overall_error / (perpendicular_total + proportional_total)
            print(f'[EVALUATE] overall error*:\t\t{error3:0.5f}\t(over {total_samples} samples)')

        sim_accuracy1 = 100 * true_similarity1 / proportional_total
        sim_accuracy2 = 100 * true_similarity2 / proportional_total
        sim_accuracy3 = 100 * true_similarity3 / proportional_total
        print(f'[EVALUATE] similarity acc:\t\t{sim_accuracy1:0.4f}\t(w.r.t threshold: {similarity_threshold1}, over {proportional_total} samples)')
        print(f'[EVALUATE] similarity acc:\t\t{sim_accuracy2:0.4f}\t(w.r.t threshold: {similarity_threshold2}, over {proportional_total} samples)')
        print(f'[EVALUATE] similarity acc:\t\t{sim_accuracy3:0.4f}\t(w.r.t threshold: {similarity_threshold3}, over {proportional_total} samples)')

        false_ratio1 = 100 * false_similarity1 / perpendicular_total
        false_ratio2 = 100 * false_similarity2 / perpendicular_total
        false_ratio3 = 100 * false_similarity3 / perpendicular_total
        print(f'[EVALUATE] (sim) false pos:\t\t{false_ratio1:0.4f}\t(w.r.t threshold: {similarity_threshold1}, over {perpendicular_total} samples)')
        print(f'[EVALUATE] (sim) false pos:\t\t{false_ratio2:0.4f}\t(w.r.t threshold: {similarity_threshold2}, over {perpendicular_total} samples)')
        print(f'[EVALUATE] (sim) false pos:\t\t{false_ratio3:0.4f}\t(w.r.t threshold: {similarity_threshold3}, over {perpendicular_total} samples)')

        accuracy1 = 100 * correct1 / total_samples
        accuracy2 = 100 * correct2 / total_samples
        accuracy3 = 100 * correct3 / total_samples
        print(f'[EVALUATE] overall acc: \t\t{accuracy1:0.4f}\t(w.r.t threshold: {similarity_threshold1}, over {total_samples} samples)')
        print(f'[EVALUATE] overall acc: \t\t{accuracy2:0.4f}\t(w.r.t threshold: {similarity_threshold2}, over {total_samples} samples)')
        print(f'[EVALUATE] overall acc: \t\t{accuracy3:0.4f}\t(w.r.t threshold: {similarity_threshold3}, over {total_samples} samples)')
        print(f'[EVALUATE] perpendicular / total:\t{(100 * perpendicular_total / total_samples):0.4f}')

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
    NUM_BATCHES = -1
    NUM_BATCHES = 200
    CFG = CFG_V7
    CFG['batch_size'] = BATCH_SIZE
    CFG['device'] = DEVICE

    nlp = load_spacy()

    # checkpoint1 = 'tmp/checkpoints/v13/epoch21_encoder1'
    # checkpoint2 = 'tmp/checkpoints/v13/epoch21_encoder2'
    # checkpoint1 = 'tmp/finetuned/iterations/v3_epoch69_iter0'
    checkpoint1 = 'tmp/checkpoints/batches/v14/epoch22_batch170000_encoder1'
    checkpoint2 = 'tmp/checkpoints/batches/v14/epoch22_batch170000_encoder2'

    dataset = 'processed-quora-duplicated-questions-train.csv'
    print(f'[INFO] evaluating dataset: {dataset}')

    dataloader = create_evaluating_dataloader(dataset, DEVICE, BATCH_SIZE)

    evaluate = evaluate_fn(SentenceEmbeddingV7, CFG, nlp, 
                           dataloader, num_batches=NUM_BATCHES)

    print('[INFO] checkpoint: ', checkpoint1)
    evaluate(checkpoint1)
    print('\n')

    print('[INFO] checkpoint: ', checkpoint2)
    evaluate(checkpoint2)
