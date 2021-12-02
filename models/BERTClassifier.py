import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForSequenceClassification

class BERTModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1, return_dict=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        y = self.sigmoid(outputs.logits).squeeze()
        return y