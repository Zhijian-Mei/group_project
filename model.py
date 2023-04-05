
from torch import nn

class RobertaMLP(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model
        self.token_to_character = nn.Linear(512,2048)
        self.output = nn.ModuleList([
            nn.Linear(config.hidden_size,config.hidden_size*2),
            nn.Linear(config.hidden_size*2,2)
                                    ])


    def forward(self,text):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state
        print(x.shape)
        x = self.token_to_character(x)
        print(x.shape)
        for module in self.output:
            x = module(x)
        return x