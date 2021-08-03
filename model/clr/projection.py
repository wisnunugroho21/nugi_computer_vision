import torch.nn as nn

class Projection(nn.Module):
    def __init__(self):
      super(Projection, self).__init__()

      self.nn_layer   = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 16)
      )

    def forward(self, res, detach = False):      
      if detach:
        return self.nn_layer(res).detach()
      else:
        return self.nn_layer(res)