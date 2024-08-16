# src/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

model_name = 'distilbert-base-uncased'
distilbert_model = DistilBertModel.from_pretrained(model_name)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        super(QNetwork, self).__init__()
        self.distilbert = distilbert_model
        self.fc1 = nn.Linear(distilbert_model.config.hidden_size * state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, input_ids, attention_mask):
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = distilbert_out.last_hidden_state
        reshaped_hidden_states = last_hidden_state.view(last_hidden_state.shape[0], -1)
        fc1_out = F.relu(self.fc1(reshaped_hidden_states))
        fc2_out = F.relu(self.fc2(fc1_out))
        return self.fc3(fc2_out)
