import torch
from torch import nn

class RobertaMLP(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model
        self.up = nn.Linear(256,1024)
        self.cls = nn.Linear(config.hidden_size,2)


    def forward(self,text):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        x = self.up(x)
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        x = self.cls(x)

        return x