
from torch import nn

class RobertaMLP(nn.Module):
    def __init__(self,Roberta_model,config):
        super().__init__()
        self.model = Roberta_model
        self.output = nn.ModuleList([
            nn.Linear(config.hidden_size,config.hidden_size*2),
            nn.Linear(config.hidden_size*2,2)
                                    ]
                                    )


    def forward(self,text):
        x = self.model(text['input_ids'],text['attention_mask']).last_hidden_state
        output = self.output(x)
        return output