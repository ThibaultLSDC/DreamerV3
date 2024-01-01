import torch
from torch.distributions import Independent, OneHotCategorical


class ReshapeCategorical:
    def __init__(self, logits, num_dists):
        self.logits = logits.reshape(logits.shape[0], num_dists, -1)
        self.num_dists = num_dists
        self.dist = Independent(OneHotCategorical(logits=self.logits), 1)

    def sample(self):
        sample = self.dist.sample()
        probs = self.dist.probs
        return sample + probs - probs.detach()
    
    def log_prob(self, x):
        return self.dist.log_prob(x)
    
    def entropy(self):
        return self.dist.entropy()
