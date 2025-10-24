import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Dict


def simple_plot(x: Union[np.ndarray, list], y: Union[np.ndarray, list, Dict[str, np.ndarray], Dict[str, list]],
                fig_file: str, title: str, xlabel: str, ylabel: str, marker: str = None):
    plt.figure(figsize=(10, 6))
    if isinstance(y, dict):
        for key_label, values in y.items():
            if marker is not None:
                plt.plot(x, values, marker=marker, label=key_label)
            else:
                plt.plot(x, y, label=key_label)
    else:
        if marker is not None:
            plt.plot(x, y, marker=marker)
        else:
            plt.plot(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if isinstance(y, dict):
        plt.legend()
    plt.savefig(fig_file)
    plt.close()
