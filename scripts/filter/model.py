import math, torch
import torch.nn as nn

from transformers import BertModel


class Classifier(nn.Module):

    def __init__(self, model_name, dropout_rate=0.5, init_value=None):

        super(Classifier, self).__init__()

        self.model = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(self.model.config.hidden_size, 512)
        self.out_proj = nn.Linear(512, 1)


    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.dropout(outputs.pooler_output)
        x = torch.nn.functional.gelu(self.dense(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        return x