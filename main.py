import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_util import get_data, ToxicDataset
from torch import nn, cuda
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaConfig, AutoTokenizer, RobertaForTokenClassification, AutoModel
from transformers import AutoConfig, AutoModelForTokenClassification
from model import *
from evaluation import f1



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--freeze', type=int, default=0)
    args = parser.parse_args()
    return args


def get_token_labal(input_encoding, label, max_length):
    attention_mask = input_encoding['attention_mask']
    golden_labels = []
    for j in range(input_encoding['input_ids'].shape[0]):
        label_for_token = [-100 for _ in range(max_length)]
        for k in range(max_length):
            if attention_mask[j][k] == 1:
                label_for_token[k] = 0
            else:
                break
            if input_encoding.token_to_chars(j, k) is None:
                label_for_token[k] = -100
                continue
            start, end = input_encoding.token_to_chars(j, k)
            for position in label[j]:
                if position == -1:
                    break
                if start <= position < end:
                    label_for_token[k] = 1
                    break
        golden_labels.append(label_for_token)
    return golden_labels


def get_char_label(input_encoding, label, text_length, max_length=1024):
    golden_labels = []
    for j in range(input_encoding['input_ids'].shape[0]):
        label_for_char = [-100 for _ in range(max_length)]
        for k in range(max_length):
            if k < text_length[j]:
                label_for_char[k] = 0
        for position in label[j]:
            label_for_char[int(position)] = 1
        golden_labels.append(label_for_char)
    return golden_labels


if __name__ == '__main__':
    args = get_args()
    print(args)
    gpu = args.gpu
    device = torch.device(f'cuda:{gpu}' if cuda.is_available() else 'cpu')

    train = pd.read_csv('data/tsd_train.csv')
    eval = pd.read_csv('data/tsd_trial.csv')
    test = pd.read_csv('data/tsd_test.csv')
    # print(train['spans'][1])
    # print(train['text'][1][train['spans'][1][0]:train['spans'][1][-1]+1])

    print('loading train data')
    trainSet = get_data(train)
    print('loading eval data')
    evalSet = get_data(eval)
    # print('loading test data')
    # testSet = get_data(test,mode='eval')
    # max_length = 0
    # for i in range(len(trainSet)):
    #     if len(trainSet['text'][i].split()) > max_length:
    #         max_length = len(trainSet['text'][i].split(' '))
    # for i in range(len(evalSet)):
    #     if len(evalSet['text'][i].split()) > max_length:
    #         max_length = len(evalSet['text'][i].split(' '))
    # for i in range(len(testSet)):
    #     if len(testSet['text'][i].split()) > max_length:
    #         max_length = len(testSet['text'][i].split(' '))
    # print(max_length)
    # quit()
    model_name = args.model
    print(f'Backbone model name: {model_name}')
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    internal_model = AutoModel.from_pretrained(model_name).to(device)
    if args.freeze == 1:
        for param in internal_model.parameters():
            param.requires_grad = False
    model = RobertaMLP_token(internal_model, config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(inputs['input_ids'].shape)
    # outputs = model(**inputs)
    #
    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape)
    max_length = 256
    trainSet = ToxicDataset(trainSet, tokenizer, max_length)
    evalSet = ToxicDataset(evalSet, tokenizer, max_length)
    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    train_loader = DataLoader(trainSet, batch_size=train_batch_size, shuffle=False)
    eval_loader = DataLoader(evalSet, batch_size=eval_batch_size)

    print('loading test data')
    testSet = get_data(test)
    testSet = ToxicDataset(testSet, tokenizer, max_length)
    test_loader = DataLoader(testSet, batch_size=eval_batch_size)

    epoch = 20
    global_step = 0
    best_f1 = 0

    for e in range(epoch):
        model.train()
        for i in tqdm(train_loader,mininterval=200):
            text, label, text_length = i[0], i[1].to(device), i[2]
            input_encoding = tokenizer.batch_encode_plus(
                text,
                max_length=max_length,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            golden_labels = get_token_labal(input_encoding, label, max_length)
            # golden_labels = get_char_label(input_encoding,label,text_length)
            # for j in range(len(golden_labels)):
            #     print(golden_labels[j])
            #     print(label[j])
            # quit()
            golden_labels = torch.LongTensor(golden_labels).to(device)
            logits, loss = model(input_encoding, golden_labels)

            # print(logits.argmax(-1).cpu().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                print('loss: ', loss.item())

        f1score = 0
        count = 0
        model.eval()
        for i in tqdm(test_loader,mininterval=200):
            text, label, _ = i[0], i[1], i[2]
            input_encoding = tokenizer.batch_encode_plus(
                text,
                max_length=max_length,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            logits = model(input_encoding)
            predicted_token_class_ids = logits.argmax(-1)
            label = label.tolist()
            predicted_labels = []
            for j in range(len(label)):
                label[j] = [int(it) for it in label[j] if it != -1]
            for j in range(input_encoding['input_ids'].shape[0]):
                label_for_char = []
                for k in range(1, max_length):
                    if predicted_token_class_ids[j][k] == 1 and input_encoding['attention_mask'][j][k] == 1:
                        start, end = input_encoding.token_to_chars(j, k)
                        for position in range(start, end):
                            label_for_char.append(position)
                predicted_labels.append(label_for_char)
            # if len(predicted_labels) != 0:
            #     print(predicted_labels)
            for j in range(len(predicted_labels)):
                f1score += f1(predicted_labels[j], label[j])
                count += 1

        f1score = f1score / count
        print(f'f1_score: {f1score} at epoch {e}')
        torch.save({'model': model.state_dict()}, f"checkpoint/{model_name}_epoch{e}_{'freeze' if args.freeze == 1 else 'unfreeze'}.pt")
        if f1score > best_f1:
            best_f1 = f1score
            torch.save({'model': model.state_dict()},
                       f"checkpoint/best_{model_name}_epoch{e}_f1:{round(best_f1, 3)}_{'freeze' if args.freeze == 1 else 'unfreeze'}.pt")
            print('saving better checkpoint')
