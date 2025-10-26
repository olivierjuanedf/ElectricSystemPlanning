from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Dict

from utils.dates import set_temporal_period_str


def set_temporal_period_title(min_date: datetime, max_date: datetime) -> str:
    print_year = max_date.year > min_date.year
    return set_temporal_period_str(min_date=min_date, max_date=max_date, print_year=print_year)


def simple_plot(x: Union[np.ndarray, list], y: Union[np.ndarray, list, Dict[str, np.ndarray], Dict[str, list]],
                fig_file: str, title: str, xlabel: str, ylabel: str, marker: str = None,
                with_curve_labels: bool = True):
    plt.figure(figsize=(10, 6))
    # TODO: merge all cases in a unique call de plt.plot
    if isinstance(y, dict):
        for key_label, values in y.items():
            current_label = key_label if with_curve_labels else None
            plt.plot(x, values, marker=marker, label=current_label)
    else:
        plt.plot(x, y, marker=marker)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if isinstance(y, dict) and with_curve_labels:
        plt.legend()
    plt.savefig(fig_file)
    plt.close()
