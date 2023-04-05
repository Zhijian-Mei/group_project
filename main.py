import pandas as pd
import torch
from data_util import get_data,ToxicDataset
from torch import nn,cuda
from torch.utils.data import DataLoader
from transformers import RobertaModel,RobertaConfig,AutoTokenizer
from model import RobertaMLP
device = torch.device('cuda:0' if cuda.is_available() else 'cpu')


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
testSet = get_data(test)
# max_length = 0
# for i in range(len(trainSet)):
#     if len(trainSet['text'][i].split()) > max_length:
#         max_length = len(trainSet['text'][i].split())
# for i in range(len(evalSet)):
#     if len(evalSet['text'][i].split()) > max_length:
#         max_length = len(evalSet['text'][i].split())
# for i in range(len(testSet)):
#     if len(testSet['text'][i].split()) > max_length:
#         max_length = len(testSet['text'][i].split())
# print(max_length)
config = RobertaConfig()
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
Roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
model = RobertaMLP(Roberta_model,config).to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs['input_ids'].shape)
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
trainSet = ToxicDataset(trainSet,tokenizer)
train_loader = DataLoader(trainSet,batch_size=8,shuffle=False)

model.train()

for i in train_loader:
    text,label = i[0].to(device),i[1].to(device).to(torch.long)
    output = model(text)
    loss = loss_f(torch.reshape(output,(output.shape[0],output.shape[2],output.shape[1])),label)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
