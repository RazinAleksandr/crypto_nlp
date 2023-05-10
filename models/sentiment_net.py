import torch
import torch.nn as nn
from typing import Type, Tuple

from transformers import logging; logging.set_verbosity_error()  # Ignore warning on model loading.
from transformers import BertModel


class TextClassifier(nn.Module):
    def __init__(self, bert_model: Type[BertModel], model_name: str, num_classes: int, n_inputs: int = 1024):
        super(TextClassifier, self).__init__()

        # Load the BERT model
        self.bert = bert_model.from_pretrained(model_name)

        # Define the layers for the classification network
        self.fc1 = nn.Linear(n_inputs + 300, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, num_classes)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, bert_inputs: Tuple[torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
        # Process the input text with BERT
        bert_outputs = self.bert(*bert_inputs)[0][:, 0, :]

        # Concatenate the BERT and FastText embeddings
        concatenated = torch.cat((bert_outputs, embeddings), dim=1)
        
        # Feed the concatenated embeddings through the classification network
        out = self.fc1(concatenated)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        
        return out
