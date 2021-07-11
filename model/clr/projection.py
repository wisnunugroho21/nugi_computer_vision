import torch.nn as nn

class Projection(nn.Module):
    def __init__(self):
      super(Projection, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 64)
      )

    def forward(self, res, detach = False):      
      if detach:
        return self.nn_layer(res).detach()
      else:
        return self.nn_layer(res)