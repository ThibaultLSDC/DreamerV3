import torch
import torch.nn as nn
import torch.nn.functional as F

from DreamerV3.models.nets import MLP, Linear
from DreamerV3.models.rssm import RSSM
from DreamerV3.models.dists import ReshapeCategorical


class WorldModel(nn.Module):
    def __init__(self,
                encoder,
                decoder,
                rssm_config,
                mlp_config,
                loss_scales={
                        'rec_loss': 1.0,
                        'reward_loss': 0.5,
                        'continue_loss': 0.2,
                        'kl_loss': 0.1,},
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss_scales = loss_scales

        self.rssm = RSSM(**rssm_config)
        self.reward_net = nn.Sequential(MLP(**mlp_config),
                                        Linear(mlp_config['out_dim'], 1))
        self.continue_net = nn.Sequential(MLP(**mlp_config),
                                        Linear(mlp_config['out_dim'], 1),
                                        nn.Sigmoid())


    def learn(self, state, action, reward, done, first, optim):
        """
        state: t -> t+T+1
        action: t -> t+T+1
        reward: t -> t+T+1
        done: t -> t+T+1
        first: t -> t+T+1
        """
        obs = self.encoder(state)

        post = self.rssm.observe(obs, action, first[:, :, None])
        start = {k: v[:, 0] for k, v in post.items()}
        prior = self.rssm.imagine(action, start)

        losses = self.get_rec_loss(state, post, reward, done)
        losses = losses | self.get_kl_loss(post, prior)
        loss = sum([v*self.loss_scales[k] for k, v in losses.items()])
        optim.zero_grad()
        loss.backward()
        optim.step()
        return losses

    def get_rec_loss(self, state, post, reward, done):
        x = torch.cat([post['stoch'], post['deter']], dim=-1)
        pred_obs = self.decoder(x)
        pred_reward = self.reward_net(x).squeeze(-1)
        pred_continue = self.continue_net(x).squeeze(-1)
        rec_loss = F.mse_loss(pred_obs, state)
        reward_loss = F.mse_loss(pred_reward, reward)
        continue_loss = F.binary_cross_entropy(pred_continue, 1-done.float())
        return {'rec_loss': rec_loss,
                'reward_loss': reward_loss,
                'continue_loss': continue_loss,}

    def get_kl_loss(self, post, prior):
        dist_prior = self.rssm.get_dist(prior['logits'])
        dist_post = self.rssm.get_dist(post['logits'])
        kl_loss = dist_post.kl(dist_prior).mean()
        return {'kl_loss': kl_loss}



if __name__ == '__main__':
    rssm_config = {
        'action_dim': 6,
        'hidden_dim': 256,
        'rec_dim': 256,
        'stochastic_dim': 32,
        'stochastic_size': 32,
        'learned_initial_state': False,
    }
    mlp_config = {
        'in_dim': 1024+256,
        'out_dim': 64,
        'layers': 1,
        'lin_out': False,
        'zero_last': False,
    }
    encoder = MLP(32, 256, 1, False, False)
    decoder = MLP(1024+256, 32, 1, False, False)
    wm = WorldModel(encoder, decoder, rssm_config, mlp_config)

    state = torch.randn(32, 16, 32)
    action = torch.randn(32, 16, 6)
    reward = torch.randn(32, 16, 1)
    done = torch.zeros(32, 16, 1)
    first = torch.zeros(32, 16, 1, dtype=torch.bool)
    losses = wm.learn(state, action, reward, done, first)
    print(losses)