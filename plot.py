from typing import Callable, Optional

import plotly.graph_objects as go
import torch


def plot_landscape(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    resolution: int = 100,
    lim: int = 3,
    height: int = 900,
    landscape_opacity: float = 1.0,
    title: Optional[str] = None,
    autoshow: bool = False,
    **kwargs
) -> go.Figure:
    """Plot the landscape defined by the function `fn`.

    Creates a domain grid $x,y \in R^2$ with $x \in [-lim, lim]$ and
    $y \in [-lim, lim]. The number of points in this grid is resolution**2.
    """
    xx = torch.linspace(-lim, lim, resolution)
    yy = torch.linspace(-lim, lim, resolution)

    yy = yy.repeat(yy.shape[0], 1)
    xx = xx.unsqueeze(-1).repeat(1, xx.shape[0])
    meshgrid = torch.stack((xx, yy), dim=-1)
    zz = fn(meshgrid, **kwargs)

    xx = xx.cpu().detach()
    yy = yy.cpu().detach()
    zz = zz.cpu().detach()

    fig = go.Figure(
        data=[
            go.Surface(
                z=zz,
                x=xx,
                y=yy,
                opacity=landscape_opacity,
                cmid=0,
                colorscale="Viridis",
            ),
        ],
    )
    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="lightgray", project_z=True
        )
    )
    fig.update_layout(
        height=height,
    )

    if autoshow:
        fig.show()
    return fig


def plot_points_over_landscape(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    points: (float, float) = None,
    resolution: int = 100,
    lim: int = 3,
    landscape_opacity: float = 1.0,
    height: int = 900,
    title: Optional[str] = None,
    autoshow: bool = False,
) -> go.Figure:
    """Plot a point over the landascape defined by the cunction `fn`

    :param fn: an universal function $R^2 -> R$
    :param points: tensor of shape [..., 3]
    :param title: the title of the plots, if None defaults to  the fn name
    :param autoshow: if True, calls fig.show() before returning the figure

    :retuns: the figure that contains the plot
    """
    points = torch.as_tensor(points)
    fig = plot_landscape(
        fn,
        resolution=resolution,
        lim=lim,
        height=height,
        landscape_opacity=landscape_opacity,
        title=title,
    )

    # Create starting path
    x_points = points[..., 0]
    y_points = points[..., 1]
    z_points = points[..., 2]

    for point in points:
        fig.add_trace(
            go.Scatter3d(
                visible=True,
                showlegend=False,
                mode="markers",
                marker=dict(
                    size=6,
                    color="black",
                    symbol="circle",
                    # opacity=0.7,
                ),
                x=x_points,
                y=y_points,
                z=z_points,
            )
        )

    if autoshow:
        fig.show()

    return fig
