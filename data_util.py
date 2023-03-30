from ast import literal_eval
from torch.utils.data import Dataset
from tqdm import trange

class ToxicDataset(Dataset):
    def __init__(self,df,tokenizer):
        self.text = df['text']
        print(self.text)
        self.label = df['spans']
        self.tokenizer = tokenizer
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
        return text,self.label[idx]

def get_data(df):
    df["spans"] = df.spans.apply(literal_eval)
    for i in trange(len(df)):
        text = df['text'][i]
        spans = df['spans'][i]
        # label = [0 for _ in range(len(text))]
        label = [0 for _ in range(2048)]
        for toxic_position in spans:
            label[toxic_position] = 1
        df['spans'][i] = label
    return df