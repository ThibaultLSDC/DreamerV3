import torch
import torch.nn as nn
import torch.distributions as td

from DreamerV3.models import Linear, GRU


class RSSM(nn.Module):
    def __init__(self,
        action_dim,
        obs_dim,
        hidden_dim,
        stoch_size,
        stoch_categoricals,
        dim,
        layers,
        ) -> None:
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.stoch_size = stoch_size
        self.stoch_categoricals = stoch_categoricals
        self.dim = dim
        self.layers = layers
        self.stoch_dim = stoch_size * stoch_categoricals

        self.encoder = Linear(
            input_dim=obs_dim + hidden_dim,
            output_dim=self.stoch_dim,
            dim=dim,
            hidden_layers=layers,
        )
        self.decoder = Linear(
            input_dim=hidden_dim+self.stoch_dim,
            output_dim=obs_dim,
            dim=dim,
            hidden_layers=layers,
        )
        self.sequence_model = GRU(
            input_dim=self.stoch_dim+action_dim,
            hidden_dim=hidden_dim,
        )
        self.dynamics_model = Linear(
            input_dim=hidden_dim,
            output_dim=self.stoch_dim,
            dim=dim,
            hidden_layers=layers,
        )
        self.continue_model = Linear(
            input_dim=hidden_dim+self.stoch_dim,
            output_dim=1,
            dim=128,
            hidden_layers=layers,
        )
        self.reward_model = Linear(
            input_dim=hidden_dim+self.stoch_dim,
            output_dim=1,
            dim=128,
            hidden_layers=layers,
        )

    def get_categorical_sample(self, logits):
        """
        Takes in logits for a categorical distrib, reshapes and returns a sample and the log_prob
        """
        shape = logits.shape
        # N * d * c
        logits = torch.reshape(logits, shape[:-1] + (self.stoch_categoricals, self.stoch_size))
        probs = logits.softmax(dim=-1)
        dist = td.OneHotCategorical(probs=probs)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        sample += log_prob.exp() - log_prob.exp().detach()
        return torch.flatten(sample, start_dim=-2, end_dim=-1), torch.flatten(log_prob, start_dim=-2, end_dim=-1)

    def encode(self, x_t, h_t):
        """
        Maps the true state x_t and the hidden state h_t to the stochastic state z_t
        :param x_t: true state given by the env
        :param h_t: hidden state given by the sequence model
        :return z_t: stochasic state
        :return probs: stochasic state's log probability
        """
        logits = self.encoder(torch.cat([x_t, h_t], dim=-1))
        return self.get_categorical_sample(logits)

    def sequence_step(self, h_t, z_t, a_t):
        """
        Predicts the next hidden states h_t+1 given the current hidden, stochastic states and action
        :param h_t: current hidden state
        :param z_t: current stochastic state
        :param a_t: current action
        :return: next hidden state
        """
        return self.sequence_model(
            torch.cat([z_t, a_t], dim=-1),
            h_t,
        )
    
    def dynamics(self, h_t):
        """
        Predicts a prior estimation of the stochastic state given the hidden state
        :param h_t: hidden sequence state
        :return: stochastic state sample, log_probs
        """
        logits =  self.dynamics_model(h_t)
        return self.get_categorical_sample(logits)

    def decode(self, h_t, z_t):
        """
        Maps the hidden (h_t) and stochastic (z_t) states to the true state x_t
        :param h_t: hidden state given by the sequence model
        :param z_t: stochastic state given by the model
        :return x_t: true state estimate
        """
        return self.decoder(torch.cat([h_t, z_t], dim=-1))

    def pred_continue(self, h_t, z_t):
        """
        Estimates the continuation of the episode (1-termination)
        :param h_t: hidden state given by the sequence model
        :param z_t: stochastic state given by the model
        :return c_t: continuation at step t
        """
        return self.continue_model(torch.cat([h_t, z_t], dim=-1))

    def pred_reward(self, h_t, z_t):
        """
        Estimates the transition reward
        :param h_t: hidden state given by the sequence model
        :param z_t: stochastic state given by the model
        :return r_t: reward estimate at step t
        """
        return self.reward_model(torch.cat([h_t, z_t], dim=-1))
    
    def imagine_one(self, actor, h_t, z_t):
        """
        Imagines the next state based on an actor's reaction
        :param actor: actor class, that has a .act() method that returns an action given hidden and stoch states
        :param h_t: hidden state given by the sequence model
        :param z_t: stochastic state given by the model

        :return next_hidden_state: the imaginated next hidden state given the actor's behavior
        :return next_stoch_state: the imaginated next stochastic state given the actor's behavior
        :return action: the action given out by the actor for that precise state
        """
        action = actor.act(torch.cat((h_t, z_t), dim=-1))
        next_hidden_state = self.sequence_step(h_t, z_t, action)
        next_stoch_state = self.dynamics(next_hidden_state)

        return next_hidden_state, next_stoch_state, action
    
    def imagine_n(self, actor, h_t, z_t, n: int):
        """
        Imagines the n next states based on an actor's reaction
        :param actor: actor class, that has a .act() method that returns an action given hidden and stoch states
        :param h_t: starting hidden state given by the sequence model
        :param z_t: starting stochastic state given by the model
        
        :return hidden_states: the imaginated sequence of hidden states given the actor's behavior
        :return stoch_states: the imaginated sequence of stochastic states given the actor's behavior
        :return actions: the sequence of actions given out by the actor for that sequence
        """
        h_tmp = h_t.clone()
        z_tmp = z_t.clone()

        actions = []
        hidden_states = []
        stoch_states = []

        for step in range(n):
            next_hidden_state, next_stoch_state, action = self.imagine_one(actor, h_tmp, z_tmp)
            actions.append(action)
            hidden_states.append(next_hidden_state)
            stoch_states.append(next_stoch_state)

            h_tmp = next_hidden_state
            z_tmp = next_stoch_state
        
        return hidden_states, stoch_states, actions
