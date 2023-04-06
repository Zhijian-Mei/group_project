import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_util import get_data, ToxicDataset
from torch import nn, cuda
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaConfig, AutoTokenizer, RobertaForTokenClassification
from transformers import AutoConfig, AutoModelForTokenClassification
from model import RobertaMLP
from evaluation import f1

device = torch.device('cuda:7' if cuda.is_available() else 'cpu')

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
config = RobertaConfig()
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = RobertaForTokenClassification.from_pretrained('roberta-base').to(device)
# model = RobertaMLP(model, config).to(device)
loss_f = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs['input_ids'].shape)
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
max_length = 256
trainSet = ToxicDataset(trainSet, tokenizer, max_length)
evalSet = ToxicDataset(evalSet, tokenizer)
train_batch_size = 4
eval_batch_size = 2
train_loader = DataLoader(trainSet, batch_size=train_batch_size, shuffle=False)
eval_loader = DataLoader(evalSet, batch_size=eval_batch_size)

epoch = 10
global_step = 0
labels_to_ids = {'T':1,'NT':0}
ids_to_labels = {1:'T',0:'NT'}
for e in range(epoch):
    model.train()
    for i in tqdm(train_loader):
        text, label = i[0], i[1].to(device)
        input_encoding = tokenizer.batch_encode_plus(
            text,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
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
        # for j in range(len(golden_labels)):
        #     print(golden_labels[j])
        #     print(input_encoding.tokens(j))
        # quit()
        golden_labels = torch.LongTensor(golden_labels).to(device)
        output = model(**input_encoding,labels=golden_labels)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step % 100 == 0:
            print('loss: ', loss.item())
            break


    f1score = 0
    count = 0
    model.eval()
    for i in tqdm(eval_loader):
        text, label = i[0], i[1]
        input_encoding = tokenizer.batch_encode_plus(
            "Damn, a whole family. Sad indeed.",
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        logits = model(**input_encoding).logits
        predicted_token_class_ids = logits.argmax(-1)
        print(predicted_token_class_ids)
        quit()
        predicted_labels = []
        for j in range(input_encoding['input_ids'].shape[0]):
            label_for_char = []
            for k in range(1, max_length):
                if predicted_token_class_ids[j][k] == 1:
                    start, end = input_encoding.token_to_chars(j, k)
                    for position in range(start, end):
                        label_for_char.append(position)
            predicted_labels.append(label_for_char)

        print(predicted_labels)
        for i in range(len(predicted_labels)):
            f1score += f1(predicted_labels[i], label[i])
            count += 1

    f1score = f1score / count
    print(f'f1_score: {f1score} at epoch {e}')
