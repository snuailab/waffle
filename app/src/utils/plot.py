import matplotlib

matplotlib.rc(
    "font",
    **{
        "family": "serif",
        "size": 16,
    }
)

import numpy as np
from matplotlib import pyplot as plt

colors = [
    [0.12156863, 0.46666667, 0.70588235],
    [1.0, 0.49803922, 0.05490196],
    [0.17254902, 0.62745098, 0.17254902],
    [0.45882353, 0.41960784, 0.69411765],
] + np.random.rand(1000, 3).tolist()


def plot_bar(
    x: list,
    y: list,
    names: list = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (10, 5),
    legend: bool = True,
):
    fig = plt.figure(figsize=figsize)
    plt.bar(x, y, label=names, color=colors[: len(x)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if names:
        plt.xticks(x, names)
    if legend:
        plt.legend()

    return fig
