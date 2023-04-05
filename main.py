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

device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

train = pd.read_csv('data/tsd_train.csv')
eval = pd.read_csv('data/tsd_trial.csv')
test = pd.read_csv('data/tsd_test.csv')
# print(train['spans'][1])
# print(train['text'][1][train['spans'][1][0]:train['spans'][1][-1]+1])

print('loading train data')
trainSet = get_data(train)
print('loading eval data')
evalSet = get_data(eval, mode='eval')
# print('loading test data')
# testSet = get_data(test)
# max_length = 0
# for i in range(len(trainSet)):
#     if len(trainSet['text'][i].split()) > max_length:
#         max_length = len(trainSet['text'][i])
# for i in range(len(evalSet)):
#     if len(evalSet['text'][i].split()) > max_length:
#         max_length = len(evalSet['text'][i])
# for i in range(len(testSet)):
#     if len(testSet['text'][i].split()) > max_length:
#         max_length = len(testSet['text'][i])
# print(max_length)
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
trainSet = ToxicDataset(trainSet, tokenizer)
evalSet = ToxicDataset(evalSet, tokenizer)

train_loader = DataLoader(trainSet, batch_size=2, shuffle=False)
eval_loader = DataLoader(evalSet, batch_size=1)

epoch = 10
global_step = 0
for e in range(epoch):
    model.train()
    for i in tqdm(train_loader):
        text, label = i[0].to(device), i[1].to(device)
        output = model(text)
        loss = loss_f(output, label)
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
        text, label = i[0].to(device), i[1]
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

