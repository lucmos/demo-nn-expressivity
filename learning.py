
from typing import Callable,  Dict
import torch
import torch.nn as nn

class PointsDataset(torch.utils.data.Dataset):
  def __init__(self, x: torch.Tensor, y_true: torch.Tensor) -> None:
    super().__init__()

    self.x = x
    self.y_true = y_true

  def __len__(self) -> int:
    return self.y_true.shape[0]

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    return {                    # For the idx-th sample
        'x': self.x[idx, ...],  # the "x" are the (x, y) coordinates of the idx-th point
        'y': self.y_true[idx]   # the "y" is  the (z) coordinate of the idx-th point
    }

class MLP2D(nn.Module):
  def __init__(self,
               num_layers: int,
               hidden_dim: int,
               activation: Callable[[torch.Tensor], torch.Tensor]
               ) -> None:
    super().__init__()

    self.first_layer = nn.Linear(in_features=2,
                                 out_features=hidden_dim)

    self.layers = nn.ModuleList()  # A list of modules: automatically exposes nested parameters to optimize.
                                   # Parameters contained in a normal python list are not returned by model.parameters()
    for i in range(num_layers):
      self.layers.append(
          nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
      )
    self.activation = activation

    self.last_layer = nn.Linear(in_features=hidden_dim,
                                out_features=1)


  def forward(self, meshgrid: torch.Tensor) -> torch.Tensor:
    """
    Applies transformations to each (x, y) independently

    :param meshgrid: tensor of dimensions [..., 2], where ... means any number of dims
    """
    out = meshgrid

    out = self.first_layer(out)  # First linear layer, transforms the hidden dimensions from 2 (the coordinates) to `hidden_dim`
    for layer in self.layers:    # Apply `k` (linear, activation) layer
      out = layer(out)
      out = self.activation(out)
    out = self.last_layer(out)   # Last linear layer to bring the `hiddem_dim` features back to the 2 coordinates x, y

    return out.squeeze(-1)
