from ast import literal_eval

import torch
from torch.utils.data import Dataset
from torch import FloatTensor
from tqdm import trange
from transformers import BatchEncoding

class ToxicDataset(Dataset):
    def __init__(self,df,tokenizer):
        self.text = list(df['text'])
        self.label = list(df['spans'])
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = FloatTensor(self.label[idx])
        return text,label

def get_data(df,mode='train'):
    df["spans"] = df.spans.apply(literal_eval)
    if mode == 'train':
        for i in trange(len(df)):
            spans = df['spans'][i]
            # label = [0 for _ in range(len(text))]
            label_for_train = [-100 for _ in range(768)]
            for j in range(len(spans)):
                print(j)
                label_for_train[j] = spans[j]
            df['spans'][i] = label_for_train
    return df