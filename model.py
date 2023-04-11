import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class RobertaMLP_char(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model
        self.up = nn.Linear(256,1024)
        self.cls = nn.Linear(config.hidden_size,2)
        self.num_labels = 2

    def forward(self,text,labels=None):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        x = self.up(x)
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        logits = self.cls(x)

        loss_fct = CrossEntropyLoss(ignore_index=-100)

        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits ,loss
        return logits

class RobertaMLP_token(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model
        self.num_labels = 2
        self.up = nn.Linear(config.hidden_size,2048)
        self.down = nn.Linear(2048,self.num_labels)


    def forward(self,text,labels=None):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state
        x = self.up(x)
        logits = self.down(x)

        loss_fct = CrossEntropyLoss(ignore_index=-100)

        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits ,loss
        return logits