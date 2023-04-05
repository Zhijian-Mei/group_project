import torch
from torch import nn

class RobertaMLP(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model

        self.cls = nn.Linear(config.hidden_size,2)


    def forward(self,text):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state

        x = self.cls(x)

        return x