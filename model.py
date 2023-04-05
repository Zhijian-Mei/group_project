
from torch import nn

class RobertaMLP(nn.Module):
    def __init__(self,Roberta_model,config):
        super().init()
        self.a = Roberta_model
        self.b = nn.Linear(768,2048)