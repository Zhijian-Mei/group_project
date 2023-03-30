import pandas as pd
import torch
from data_util import get_data

from transformers import RobertaModel,RobertaConfig,AutoTokenizer
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
model = RobertaModel.from_pretrained('roberta-base')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)