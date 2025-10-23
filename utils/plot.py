import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def simple_plot(x: Union[np.ndarray, list], y: Union[np.ndarray, list], fig_file: str,
                title: str, xlabel: str, ylabel: str, marker: str = None):
    plt.figure(figsize=(10, 6))
    if marker is not None:
        plt.plot(x, y, marker=marker)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(fig_file)
    plt.close()
