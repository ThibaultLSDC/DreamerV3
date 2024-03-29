import torch
import torch.nn as nn

from .nets import Activation, Linear, MLP, GRUCell
from .dists import ReshapeCategorical


def swap(x):
    return torch.transpose(x, 0, 1)


class RSSM(nn.Module):
    def __init__(self,
                action_dim: int,
                hidden_dim: int=256,
                rec_dim: int=256,
                stochastic_dim: int=32,
                stochastic_size: int=32,
                learned_initial_state: bool=False,):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rec_dim = rec_dim
        self.stochastic_dim = stochastic_dim
        self.stochastic_size = stochastic_size
        self.latent_dim = stochastic_dim * stochastic_size

        # Encoder to hidden
        self._in = Linear(self.latent_dim + action_dim, hidden_dim)
        # Next deterministic state prediction
        self._gru = GRUCell(hidden_dim, rec_dim)
        # Prior
        self._prior = Linear(rec_dim, self.latent_dim)
        # Posterior / Dynamics
        self._post = Linear(rec_dim + hidden_dim, self.latent_dim)

        # Initial state
        if learned_initial_state:
            self._initial = nn.Parameter(torch.zeros(rec_dim, dtype=torch.float32))
        else:
            self._initial = torch.zeros(rec_dim, dtype=torch.float32)

    def initialize(self, batch_size):
        device = next(self.parameters()).device
        deter = self._initial.repeat(batch_size, 1).to(device)
        stoch = torch.zeros(batch_size, self.latent_dim, dtype=torch.float32).to(device)
        logits = torch.zeros(batch_size, self.latent_dim, dtype=torch.float32).to(device)
        return {'deter': deter, 'stoch': stoch, 'logits': logits}
    
    def get_dist(self, logits):
        return ReshapeCategorical(logits,  self.stochastic_size)
    
    def observe(self, obs, action, first, state=None):
        """
        obs: t+1 -> t+T
        action: t -> t+T-1
        first: t -> t+T-1

        posterior: t+1 -> t+T
        """
        obs = swap(obs)
        action = swap(action)
        first = swap(first)

        if state is None:
            state = self.initialize(obs.shape[1])
        posterior = {'deter': [], 'stoch': [], 'logits': []}
        for o, a, f in zip(obs, action, first):
            state = self.obs_step(state, a, o, f)
            for k, v in state.items():
                posterior[k].append(v)
        for k, v in posterior.items():
            posterior[k] = torch.stack(v, dim=1)
        return posterior
    
    def imagine(self, action, state=None):
        """
        action: t -> t+T-1

        prior: t+1 -> t+T
        """
        action = swap(action)

        if state is None:
            state = self.initialize(action.shape[0])
        prior = {'deter': [], 'stoch': [], 'logits': []}
        for a in action:
            state = self.img_step(state, a)
            for k, v in state.items():
                prior[k].append(v)
        for k, v in prior.items():
            prior[k] = torch.stack(v, dim=1)
        return prior

    def obs_step(self, state, action, obs, first):
        """
        Posterior prediction

        Args:
            state (dict): Dictionary containing all state information
                state['stoch'] (torch.Tensor): z_t-1 shape(bs, latent_dim)
                state['deter'] (torch.Tensor): h_t-1 shape(bs, rec_dim)
            action (torch.Tensor): a_t-1 shape(bs, action_dim)
            obs (torch.Tensor): o_t shape(bs, hidden_dim)
            first (torch.Tensor): first_t-1 shape(bs, 1)
        
        Returns:
            dict: Dictionary containing all state information
                state['stoch'] (torch.Tensor): z_t shape(bs, latent_dim)
                state['deter'] (torch.Tensor): h_t shape(bs, rec_dim)
                state['logits'] (torch.Tensor): logits shape(bs, latent_dim)
        """
        initial = self.initialize(state['stoch'].shape[0])
        state = {k: torch.where(first, initial[k], v) for k, v in state.items()}
        prior = self.img_step(state, action)
        x = torch.cat((prior['deter'], obs), dim=-1)
        logits = self._post(x)
        dist = self.get_dist(logits)
        stoch = dist.sample()
        deter = prior['deter']
        return {'deter': deter, 'stoch': stoch, 'logits': logits}

    def img_step(self, state, action):
        """
        Prior prediction
        
        Args:
            state (dict): Dictionary containing all state information
                state['stoch'] (torch.Tensor): z_t-1 shape(bs, latent_dim)
                state['deter'] (torch.Tensor): h_t-1 shape(bs, rec_dim)
            action (torch.Tensor): a_t-1 shape(bs, action_dim)
        
        Returns:
            dict: Dictionary containing all state information
                state['stoch'] (torch.Tensor): ~z_t shape(bs, latent_dim)
                state['deter'] (torch.Tensor): h_t shape(bs, rec_dim)
                state['logits'] (torch.Tensor): logits shape(bs, latent_dim)
        """
        x = torch.cat((state['stoch'], action), dim=-1)
        x = self._in(x)
        x = self._gru(x, state['deter'])
        logits = self._prior(x)
        dist = self.get_dist(logits)
        stoch = dist.sample()
        deter = x
        return {'deter': deter, 'stoch': stoch, 'logits': logits}
