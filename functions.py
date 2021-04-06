# @title utility functions

from typing import Callable, Union, Sequence
import math
import torch

def peaks(meshgrid: torch.Tensor) -> torch.Tensor:
  """
  "Peaks" function that has multiple local minima.

  :params meshgrid: tensor of shape [..., 2], the (x, y) coordinates
  """
  meshgrid = torch.as_tensor(meshgrid, dtype=torch.float)
  xx = meshgrid[..., 0]
  yy = meshgrid[..., 1]
  return (0.25 * (3*(1-xx)**2*torch.exp(-xx**2 - (yy+1)**2) -
                  10*(xx/5 - xx**3 - yy**5)*torch.exp(-xx**2-yy**2) -
                  1/3*torch.exp(-(xx+1)**2 - yy**2)))


def rastrigin(meshgrid: torch.Tensor, shift: int = 0) -> torch.Tensor:
  """
  "Rastrigin" function with `A = 3`
  https://en.wikipedia.org/wiki/Rastrigin_function

  :params meshgrid: tensor of shape [..., 2], the (x, y) coordinates
  """
  meshgrid = torch.as_tensor(meshgrid, dtype=torch.float)
  xx = meshgrid[..., 0]
  yy = meshgrid[..., 1]
  A = 3
  return A * 2 + (((xx - shift) ** 2 - A * torch.cos(2 * torch.tensor(math.pi, dtype=torch.float, device=xx.device) * xx))
                  +
                  ((yy - shift) ** 2 - A * torch.cos(2 * torch.tensor(math.pi, dtype=torch.float, device=xx.device) * yy)))


def rosenbrock(meshgrid: torch.Tensor) -> torch.Tensor:
  """
  "Rosenbrock" function
  https://en.wikipedia.org/wiki/Rosenbrock_function

  It has a global minimum at $(x , y) = (a, a^2) = (1, 1)$

  :params meshgrid: tensor of shape [..., 2], the (x, y) coordinates
  """
  meshgrid = torch.as_tensor(meshgrid, dtype=torch.float)
  xx = meshgrid[..., 0]
  yy = meshgrid[..., 1]

  a = 1
  b = 100
  return (a - xx) ** 2 + b * (yy - xx**2)**2


def simple_fn(meshgrid: torch.Tensor) -> torch.Tensor:
  """
  :params meshgrid: tensor of shape [..., 2], the (x, y) coordinates
  """
  meshgrid = torch.as_tensor(meshgrid, dtype=torch.float)
  xx = meshgrid[..., 0]
  yy = meshgrid[..., 1]

  output = -1/(1 + xx**2 + yy**2)

  return output

def simple_fn2(meshgrid: torch.Tensor) -> torch.Tensor:
  """
  :params meshgrid: tensor of shape [..., 2], the (x, y) coordinates
  """
  meshgrid = torch.as_tensor(meshgrid, dtype=torch.float)
  xx = meshgrid[..., 0]
  yy = meshgrid[..., 1]

  output = (1 + xx**2 + yy**2) ** (1/2)

  return output
