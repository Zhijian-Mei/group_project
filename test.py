import pandas as pd
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import RobertaModel, AutoTokenizer, RobertaConfig

from data_util import get_data, ToxicDataset
from evaluation import f1
from model import RobertaMLP_token



device = torch.device('cuda:7' if cuda.is_available() else 'cpu')



max_length = 256
config = RobertaConfig()
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base').to(device)

eval_batch_size = 8
test = pd.read_csv('data/tsd_test.csv')
print('loading test data')
testSet = get_data(test)
testSet = ToxicDataset(testSet, tokenizer,max_length,eval=True)
test_loader = DataLoader(testSet, batch_size=eval_batch_size)


checkpoint = torch.load('checkpoint/best_roberta_epoch4_f1:0.084.pt')
model = RobertaMLP_token(roberta, config).to(device)
model.load_state_dict(checkpoint['roberta'])



f1score = 0
count = 0
model.eval()
for i in test_loader:
    text, label, _ = i[0], i[1], i[2]
    input_encoding = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    logits = model(input_encoding)
    predicted_token_class_ids = logits.argmax(-1)

    predicted_labels = []
    for j in range(len(label)):
        label[j] = [it.item() for it in label[j] if it.item() != -1]
    for j in range(input_encoding['input_ids'].shape[0]):
        label_for_char = []
        for k in range(1, max_length):
            if predicted_token_class_ids[j][k] == 1 and input_encoding['attention_mask'][j][k] == 1:
                start, end = input_encoding.token_to_chars(j, k)
                for position in range(start, end):
                    label_for_char.append(position)
        predicted_labels.append(label_for_char)
    # if len(predicted_labels) != 0:
    #     print(predicted_labels)
    for j in range(len(predicted_labels)):
        print(predicted_labels[j])
        print(label[j])
        quit()
        f1score += f1(predicted_labels[j], label[j])
        count += 1

f1score = f1score / count
print(f'f1_score: {f1score}')
