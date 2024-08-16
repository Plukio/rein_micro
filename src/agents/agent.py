# src/agents/agent.py

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.models.model import QNetwork
from src.agents.replay_buffer import ReplayBuffer
from src.utils.utils import array2str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(27, action_size).to(device)
        self.qnetwork_target = QNetwork(27, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-5, betas=(0.9, 0.999))
        self.memory = ReplayBuffer(action_size, int(1e5), 512)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % 16 == 0 and len(self.memory) > 512:
            experiences = self.memory.sample()
            self.learn(experiences, 0.99)

    def act(self, state, eps):
        state_str = array2str(state)
        token = tokenizer(state_str, add_special_tokens=True, max_length=27, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = token["input_ids"].to(device)
        attention_mask = token["attention_mask"].to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(input_ids, attention_mask)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        state_str_list = [array2str(state) for state in states]
        next_state_str_list = [array2str(next_state) for next_state in next_states]

        token = tokenizer(state_str_list, add_special_tokens=True, max_length=27, truncation=True, padding='max_length', return_tensors='pt')
        next_token = tokenizer(next_state_str_list, add_special_tokens=True, max_length=27, truncation=True, padding='max_length', return_tensors='pt')

        input_ids_batch = token['input_ids'].to(device)
        attention_mask_batch = token['attention_mask'].to(device)
        next_input_ids_batch = next_token["input_ids"].to(device)
        next_attention_mask_batch = next_token["attention_mask"].to(device)

        Q_targets_next = self.qnetwork_target(next_input_ids_batch, next_attention_mask_batch).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(input_ids_batch, attention_mask_batch).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.t_step % (8 * 16) == 0:
            self.update_target_net(self.qnetwork_local, self.qnetwork_target)

    def update_target_net(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    def save(self, name):
        torch.save(self.qnetwork_local.state_dict(), f'checkpoints/model_{name}.pth')
        print('Model saved.')
