import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_util import get_data, ToxicDataset
from torch import nn, cuda
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaConfig, AutoTokenizer
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
print('loading test data')
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
Roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
model = RobertaMLP(Roberta_model, config).to(device)
loss_f = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs['input_ids'].shape)
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
max_length = 256
trainSet = ToxicDataset(trainSet, tokenizer,max_length)
evalSet = ToxicDataset(evalSet, tokenizer)
train_batch_size = 4
train_loader = DataLoader(trainSet, batch_size =train_batch_size, shuffle=False)
eval_loader = DataLoader(evalSet, batch_size=1)

epoch = 10
global_step = 0

for e in range(epoch):
    model.train()
    for i in tqdm(train_loader):
        input_encoding, label = i[0].to(device), i[1].to(device)
        print(input_encoding.shape)
        quit()
        golden_labels = []
        for j in range(train_batch_size):
            label_for_token = [[0,1] for _ in range(max_length)]
            for k in range(1,max_length):
                if input_encoding.token_to_chars(j,k) is None:
                    continue
                start,end = input_encoding.token_to_chars(j,k)
                for position in label[j]:
                    if position == -100:
                        break
                    if start <= position < end:
                        label_for_token[k] = [1,0]
                        break
            golden_labels.append(label_for_token)
        golden_labels = torch.FloatTensor(golden_labels).to(device)
        output = model(input_encoding)
        loss = loss_f(output, golden_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step+=1
        if global_step % 200 == 0:
            print('loss: ', loss.item())


    f1score = 0
    count = 0
    model.eval()
    for i in tqdm(eval_loader):
        text, label = i[0], i[1]
        input_encoding = tokenizer.batch_encode_plus(
            text,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        output = model(text)
        output = torch.max(output, dim=-1)[1][0]
        result = []
        for j in range(len(output)):
            print(output[j].item())
            if output[j].item() == 0:
                result.append(j)
        f1score += f1(result, label)
        count += 1

        print(result)
        print()
    f1score = f1score / count
    print('f1_score: ', f1score)

