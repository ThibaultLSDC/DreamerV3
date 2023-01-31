import torch
import torch.nn as nn


class V3Activation(nn.Module):
    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.SiLU(self.norm(x))


class Linear(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        dim: int=128,
        hidden_layers: int=1,
        end_activation: bool=False,
        ) -> None:
        super().__init__()

        l = [input_dim] + [dim] * hidden_layers + [output_dim]
        layers = []
        for i, o in zip(l[:-1], l[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(V3Activation((o,)))
        
        if not end_activation:
            layers = layers[:-1]

        self.core = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.core(x)


class GRU(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim,
        ) -> None:
        super().__init__()

        self.gru_cell = nn.GRU(input_dim, hidden_dim)
        self.activation = V3Activation(hidden_dim)

    def forward(self, x, h):
        return self.activation(self.gru_cell(x, h))