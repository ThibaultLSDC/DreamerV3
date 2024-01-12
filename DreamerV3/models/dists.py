import torch
from torch.distributions import Independent, OneHotCategorical
from torch.distributions.kl import kl_divergence


class ReshapeCategorical:
    def __init__(self, logits, num_dists):
        self.logits = logits.reshape(logits.shape[0], num_dists, -1)
        self.num_dists = num_dists
        self.dist = Independent(OneHotCategorical(logits=self.logits), 1)

    def sample(self):
        sample = self.dist.sample().reshape(self.logits.shape[0], -1)
        probs = self.dist.base_dist.probs.reshape(self.logits.shape[0], -1)
        return sample + probs - probs.detach()
    
    def log_prob(self, x):
        return self.dist.log_prob(x)
    
    def entropy(self):
        return self.dist.entropy()

    def kl(self, other):
        return kl_divergence(self.dist, other.dist)