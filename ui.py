import argparse
import contextlib
from typing import Callable, Dict, Mapping, Optional, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn as nn
from torch.utils.data import DataLoader

from functions import peaks, rastrigin, rosenbrock, simple_fn, simple_fn2
from plot import plot_points_over_landscape

st.set_page_config(layout="centered")
torch.manual_seed(0)

st.sidebar.header("Visualization")
plot_height = st.sidebar.slider(
    "Plot height:", min_value=100, max_value=1000, value=700, step=50
)
show_code = st.sidebar.checkbox("Show MLP code")


class PointsDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y_true: torch.Tensor) -> None:
        super().__init__()

        self.x = x
        self.y_true = y_true

    def __len__(self) -> int:
        return self.y_true.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {  # For the idx-th sample
            "x": self.x[
                idx, ...
            ],  # the "x" are the (x, y) coordinates of the idx-th point
            "y": self.y_true[idx],  # the "y" is  the (z) coordinate of the idx-th point
        }


with st.echo() if show_code else contextlib.nullcontext():

    class MLP2D(nn.Module):
        def __init__(
            self,
            num_layers: int,
            hidden_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor],
        ) -> None:
            super().__init__()

            self.first_layer = nn.Linear(in_features=2, out_features=hidden_dim)

            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                )
            self.activation = activation

            self.last_layer = nn.Linear(in_features=hidden_dim, out_features=1)

        def forward(self, meshgrid: torch.Tensor) -> torch.Tensor:
            """
            Applies the MLP to each (x, y) independently

            :param meshgrid: tensor of dimensions [..., 2]
            :returns: tensor of dimension [..., 1]
            """
            out = meshgrid

            out = self.first_layer(out)
            for layer in self.layers:
                out = self.activation(layer(out))
            out = self.last_layer(out)

            return out.squeeze(-1)


@st.cache(allow_output_mutation=True)
def get_dataloader(
    lim: int,
    fn: Callable,
    n_rand: int,
):
    x_random = torch.rand(n_rand) * lim * 2 - lim
    y_random = torch.rand(n_rand) * lim * 2 - lim

    xy_data = torch.stack((x_random, y_random), dim=-1)
    xy_groundtruth = fn(xy_data)

    points_dl = DataLoader(
        PointsDataset(x=xy_data, y_true=xy_groundtruth),
        batch_size=16,
        shuffle=True,
    )

    points = torch.cat((xy_data, xy_groundtruth[..., None]), dim=-1)
    return points_dl, points


fn_names = {
    "peaks": peaks,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "simple_fn": simple_fn,
    "simple_fn2": simple_fn2,
}


st.sidebar.header("Dataset creation")
base_fn = st.sidebar.selectbox("Select unknown function:", list(fn_names.keys()))
fn = fn_names[base_fn]

n_rand = st.sidebar.slider(
    "Number of points to sample:", min_value=1, max_value=300, value=80, step=10
)

lim = st.sidebar.slider("Domain limits:", min_value=0, max_value=50, value=3, step=1)


points_dl, points = get_dataloader(lim, fn, n_rand)

st.subheader("Sampled points")
st.plotly_chart(
    plot_points_over_landscape(fn, points, lim=lim, height=plot_height),
    use_container_width=True,
)


activation_names = {
    "relu": torch.relu,
    "leaky_relu": torch.nn.functional.leaky_relu,
    "elu": torch.nn.functional.elu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "identity": lambda x: x,
}

st.sidebar.header("Hyperparameters")
num_layers = st.sidebar.slider(
    "Number of hidden layers:", min_value=1, max_value=30, value=2
)

hidden_dim = st.sidebar.slider(
    "Hidden dimensionality:", min_value=1, max_value=512, value=16
)

activation_fn = st.sidebar.selectbox(
    "Select activation function:", list(activation_names.keys())
)
activation_fn = activation_names[activation_fn]

learning_rate = st.sidebar.number_input(
    "Select learning rate",
    min_value=0.0,
    max_value=10.0,
    value=0.01,
    step=0.00001,
    format="%.5f",
)

num_epochs = st.sidebar.slider(
    "Number of epochs:", min_value=0, max_value=1000, value=100, step=10
)


def energy(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)


model = MLP2D(num_layers=num_layers, activation=activation_fn, hidden_dim=hidden_dim)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_epochs):
    for batch in points_dl:
        x = batch["x"]
        y = batch["y"]
        y_pred = model(x)

        loss = energy(y_pred, y)

        loss.backward()
        opt.step()
        opt.zero_grad()


st.subheader("Learned function")

st.plotly_chart(
    plot_points_over_landscape(model.cpu(), points.cpu(), lim=lim, height=plot_height),
    use_container_width=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("Author: [`Luca Moschella`](https://luca.moschella.dev)")
