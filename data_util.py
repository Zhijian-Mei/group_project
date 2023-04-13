from ast import literal_eval

import torch
from torch.utils.data import Dataset
from torch import FloatTensor
from tqdm import trange
from transformers import BatchEncoding

class ToxicDataset(Dataset):
    def __init__(self,df,tokenizer,max_length=256,eval=False):
        self.eval = eval
        self.text = list(df['text'])
        self.label = list(df['spans'])
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        if self.eval:
            label = self.label[idx]
            return text,label,len(text)
        label = FloatTensor(self.label[idx])
        return text,label,len(text)

def get_data(df):
    df["spans"] = df.spans.apply(literal_eval)
    for i in trange(len(df)):
        spans = df['spans'][i]
        if len(spans) > 1024:
            print(len(spans))
            quit()
        # label = [0 for _ in range(len(text))]
        label_for_train = [-1 for _ in range(1024)]
        for j in range(len(spans)):
            label_for_train[j] = spans[j]
        df['spans'][i] = label_for_train
    return df