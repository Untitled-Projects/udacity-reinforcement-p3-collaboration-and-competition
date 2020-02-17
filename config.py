import torch

class Config:
    def __init__(self):
        self.device = 'cpu'
        self.seed = 0
        
        self.state_size = None
        self.action_size = None
        self.buffer_size = 100000
        self.batch_size = 128
        self.num_agents = 1
        self.memory = None
        self.gamma = 0.99
        self.tau = 0.02
        self.lr_actor = 0.01
        self.lr_critic = 0.01
        self.weight_decay = 0.00001
        self.update_every = 20
        self.num_updates = 1