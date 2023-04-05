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
        self.batchEncoding = None
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.tokenizer.batch_encode_plus(
            [self.text[idx]],
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.batchEncoding = BatchEncoding(text)
        print(self.batchEncoding.token_to_chars())
        quit()
        text['input_ids'] = torch.squeeze(text['input_ids'])
        label = FloatTensor(self.label[idx])
        return text,label

def get_data(df,mode='train'):
    df["spans"] = df.spans.apply(literal_eval)
    if mode == 'train':
        for i in trange(len(df)):
            spans = df['spans'][i]
            # label = [0 for _ in range(len(text))]
            label_for_train = [[0,1] for _ in range(1024)]
            for toxic_position in spans:
                label_for_train[toxic_position] = [1,0]
            df['spans'][i] = label_for_train
    return df