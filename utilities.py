import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.io import loadmat
from plotly.tools import mpl_to_plotly
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_resource
def read_data(problem: str, model: str, hyperparams_mode: str):
    return loadmat(
        f"outputs_for_website/{problem}_{model}_trainL1_{hyperparams_mode}.mat"
    )


@st.cache_data
def compute_relative_error(data: dict):
    ## Compute L1 relative error
    num_examples = data["output"].shape[0]
    diff_norms = np.linalg.norm(
        data["output"].reshape(num_examples, -1)
        - data["prediction"].reshape(num_examples, -1),
        ord=1,
        axis=1,
    )
    y_norms = np.linalg.norm(data["output"].reshape(num_examples, -1), ord=1, axis=1)

    # Check division by zero
    if np.any(y_norms <= 1e-5):
        raise ValueError("Division by zero")

    relative_diff = diff_norms / y_norms

    return relative_diff


@st.cache_data
def plot_data(
    data,
    sample_idx: int,
    ylabel: str = None,
    title: str = None,
    big_axis: bool = False,
):
    """
    data: tensor with shape (num_samples, num_points, num_points)

    sample_idx: int
        The index of the sample to be plotted.

    index_plot: int
        The index of the plot to be displayed.

    ylabel: str
        The label of the y-axis.

    title: str
        The title of the plot.

    big_axis: bool
        If True, the axis will be bigger 128x128 instead of 64x64.
    """
    plot_data = data[sample_idx]

    fig = go.Figure()
    fig = go.Figure(data=go.Heatmap(z=plot_data, colorscale="Viridis", showscale=True))

    if ylabel:
        fig.update_layout(
            yaxis_title=ylabel,
        )

    if title:
        fig.update_layout(
            title={
                "text": title,
                "x": 0.45,
                "xanchor": "center",
            },
        )

    fig.update_layout(
        xaxis={
            "title": "x",
            "tickmode": "array",
            "tickvals": [0, 63, 127] if big_axis else [0, 32, 63],
            "ticktext": [0, 0.5, 1],
        },
        yaxis={
            "tickmode": "array",
            "tickvals": [0, 63, 127] if big_axis else [0, 32, 63],
            "ticktext": [0, 0.5, 1],
        },
        font=dict(family="Arial", size=12),
        # plot_bgcolor="#f5f7fa",
        # paper_bgcolor="#ffffff",
        width=400,
        height=300,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=30,  # top margin
            b=20,  # bottom margin
        ),
    )

    return fig


@st.cache_data
def plot_error(
    data: dict,
    sample_idx: int,
    ylabel: str = None,
    title: str = None,
    big_axis: bool = False,
):
    error_data = np.abs(data["output"] - data["prediction"])

    return plot_data(
        error_data,
        sample_idx,
        ylabel=ylabel,
        title=title,
        big_axis=big_axis,
    )


@st.cache_data
def plot_histogram(data, title: str = None):

    sns.set(style="whitegrid", palette="deep")
    plt.figure(figsize=(12, 6))
    sns.histplot(data, bins=30, kde=True, color="skyblue", edgecolor="black")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Convert Matplotlib figure to Plotly figure
    plotly_fig = mpl_to_plotly(plt.gcf())
    fig = go.Figure()
    for trace in plotly_fig["data"]:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=r"Relative L<sup>1</sup> error",
        yaxis_title="Number of Samples",
        template="plotly_white",
        bargap=0.1,
        showlegend=False,
    )

    return fig


@st.cache_data
def plot_swarmplot(data, title: str = None):

    plt.figure(figsize=(4, 6))
    sns.swarmplot(x=np.zeros(len(data)), y=data, color="skyblue", size=5)

    # Convert Matplotlib figure to Plotly figure
    plotly_fig = mpl_to_plotly(plt.gcf())
    fig = go.Figure()
    for trace in plotly_fig["data"]:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis={
            "title": "Samples numerosity",
            "tickmode": "array",
            "tickvals": [],
            "ticktext": [],
        },
        yaxis_title=r"Relative L<sup>1</sup> error",
        template="plotly_white",
    )
    return fig
