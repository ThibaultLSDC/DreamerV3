import torch
import torch.nn as nn



class Activation(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.layer_norm(x))


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)
    

class MLP(nn.Module):
    def __init__(self,
                in_dim: int,
                out_dim: int=64,
                layers: int=1,
                lin_out: bool=False,
                zero_last: bool=False,):
        super().__init__()
        model = [Linear(in_dim, out_dim)]
        for _ in range(layers):
            model.append(Linear(out_dim, out_dim))
            model.append(Activation(out_dim))
        if lin_out:
            model.append(Linear(out_dim, out_dim))
        self.model = nn.Sequential(*model)
        
        if zero_last:
            self.model[-1].weight.data.zero_()
            self.model[-1].bias.data.zero_()

    def forward(self, x):
        return self.model(x)


# class RSSM(nn.Module):
