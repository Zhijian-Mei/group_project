import pandas as pd
import torch
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, AutoTokenizer, RobertaConfig, AutoConfig, AutoModel

from data_util import get_data, ToxicDataset
from evaluation import f1
from model import RobertaMLP_token
from detoxify import Detoxify


device = torch.device('cuda:7' if cuda.is_available() else 'cpu')



max_length = 256
model_name = 'bert-base-uncased'
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
internal_model = AutoModel.from_pretrained(model_name).to(device)

eval_batch_size = 16
test = pd.read_csv('data/tsd_test.csv')
print('loading test data')
testSet = get_data(test)
testSet = ToxicDataset(testSet, tokenizer,max_length)
test_loader = DataLoader(testSet, batch_size=eval_batch_size)


checkpoint = torch.load('checkpoint/best_bert-base-uncased_epoch0_f1:0.65.pt')
model = RobertaMLP_token(internal_model, config).to(device)
model.load_state_dict(checkpoint['model'])

# cls = Detoxify('original')

f1score = 0
count = 0
model.eval()
for i in tqdm(test_loader):
    text, label, _ = i[0], i[1], i[2]
    input_encoding = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    # results = cls.predict(text)['toxicity']

    logits = model(input_encoding)
    predicted_token_class_ids = logits.argmax(-1)

    predicted_labels = []
    label = label.tolist()
    for j in range(len(label)):
        label[j] = [int(it) for it in label[j] if it != -1]
    for j in range(input_encoding['input_ids'].shape[0]):
        label_for_char = []
        for k in range(1, max_length):
            if predicted_token_class_ids[j][k] == 1 and input_encoding['attention_mask'][j][k] == 1:
                start, end = input_encoding.token_to_chars(j, k)
                for position in range(start, end):
                    label_for_char.append(position)
        predicted_labels.append(label_for_char)

    for j in range(len(predicted_labels)):
        # if results[j] > 0.5:
        #     f1score += f1(predicted_labels[j], label[j])
        # else:
        #     f1score += f1([], label[j])
        f1score += f1(predicted_labels[j], label[j])
        count += 1

f1score = f1score / count
print(f'f1_score: {f1score}')
