from ast import literal_eval

import torch
from torch.utils.data import Dataset
from torch import FloatTensor
from tqdm import trange
from transformers import BatchEncoding

class ToxicDataset(Dataset):
    def __init__(self,df,tokenizer,max_length=256):
        self.text = list(df['text'])
        self.label = list(df['spans'])
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        input_encoding = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label = FloatTensor(self.label[idx])
        return input_encoding,label

def get_data(df,mode='train'):
    df["spans"] = df.spans.apply(literal_eval)
    if mode == 'train':
        for i in trange(len(df)):
            spans = df['spans'][i]
            if len(spans) > 1024:
                print(len(spans))
            # label = [0 for _ in range(len(text))]
            label_for_train = [-100 for _ in range(1024)]
            for j in range(len(spans)):
                label_for_train[j] = spans[j]
            df['spans'][i] = label_for_train
    return df