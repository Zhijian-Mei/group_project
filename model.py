import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class RobertaMLP(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model
        self.up = nn.Linear(256,1024)
        self.cls = nn.Linear(config.hidden_size,2)


    def forward(self,text,labels):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        x = self.up(x)
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        logits = self.cls(x)

        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        attention_mask = text['attention_mask']
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_fct(active_logits, active_labels)

        return logits ,loss