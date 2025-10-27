from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Dict, List

from common.constants.temporal import DAY_OF_WEEK
from utils.basic_utils import lowest_common_multiple
from utils.dates import set_temporal_period_str, add_day_exponent


def set_temporal_period_title(min_date: datetime, max_date: datetime) -> str:
    print_year = max_date.year > min_date.year
    return set_temporal_period_str(min_date=min_date, max_date=max_date, print_year=print_year)


def set_xtick_idx(min_date: datetime, max_date: datetime, delta_date: timedelta, min_delta_xticks_h: int = 1,
                  n_max_xticks: int = 15) -> List[int]:
    allowed_delta_date_xticks_h = np.array([1, 2, 4, 6, 12, 24, 7 * 24, 2 * 7 * 24, 4 * 7 * 24, 4 * 4 * 7 * 24])
    delta_date_h = int(delta_date.total_seconds() // 3600)
    delta_tot_h = int((max_date - min_date).total_seconds() // 3600)
    # set delta date between xticks (i) bigger than delta_date_h, and such that (ii) total number of xticks be smaller
    # than n_max_xticks
    delta_xticks_h_min = max(delta_date_h, min_delta_xticks_h, delta_tot_h / n_max_xticks)
    i_delta_xticks = np.where(allowed_delta_date_xticks_h >= delta_xticks_h_min)[0][0]
    delta_xticks_h = allowed_delta_date_xticks_h[i_delta_xticks]
    delta_xticks_h = lowest_common_multiple(a=delta_xticks_h, b=delta_date_h)
    n_dates = delta_tot_h // delta_date_h + 1
    return np.arange(0, n_dates, delta_xticks_h)


def set_dow_xtick_labels(idx_xticks: List[int], x_dates: List[datetime]) -> List[str]:
    new_date = None
    current_day_date = None
    i = 0
    n_xticks = len(idx_xticks)
    xtick_labels = []
    while i < n_xticks:
        # add dow only for first tick of this dow
        if new_date is None or not current_day_date == new_date:
            new_date = x_dates[idx_xticks[i] - 1]
            new_date = datetime(year=new_date.year, month=new_date.month, day=new_date.day)
            current_dow = DAY_OF_WEEK[new_date.isoweekday() - 1]
            xtick_labels[i] = f"{current_dow}\n{new_date:%H:}"
        # only hours for the other dates
        else:
            xtick_labels[i] = f"{x_dates[idx_xticks[i] - 1]:%H:}"
        i += 1
        if i < n_xticks:
            current_day_date = x_dates[idx_xticks[i] - 1]
            current_day_date = datetime(year=current_day_date.year, month=current_day_date.month,
                                        day=current_day_date.day)
    return xtick_labels


def set_month_in_letter_xtick_labels(idx_xticks: List[int], x_dates: List[datetime],
                                     add_day_exp: bool = False) -> List[str]:
    with_year_in_xticks = x_dates[-1].year > x_dates[0].year
    new_year_and_month = None
    current_year_and_month = None
    i = 0
    n_xticks = len(idx_xticks)
    xtick_labels = []
    while i < n_xticks:
        # add dow only for first tick of this dow
        if new_year_and_month is None or not current_year_and_month == new_year_and_month:
            new_date = x_dates[idx_xticks[i] - 1]
            new_year_and_month = (new_date.year, new_date.month)
            # TODO: set minimal xtick labels (H if common year, month, day...)
            if with_year_in_xticks and (current_year_and_month is None
                                        or current_year_and_month[0] > new_year_and_month[0]):
                date_fmt = '%Y %B %d'
            else:
                bob = 1
            date_str = new_date.strftime(date_fmt)
            if add_day_exp:
                date_str = add_day_exponent(date=date_str)
            xtick_labels[i] = date_str
        # only hours for the other dates
        else:
            xtick_labels[i] = f"{x_dates[idx_xticks[i] - 1]:%H:}"
        i += 1
        if i < n_xticks:
            current_date = x_dates[idx_xticks[i] - 1]
            current_year_and_month = (current_date.year, current_date.month)
    return xtick_labels


def set_date_xtick_labels(x_dates: List[datetime], min_delta_xticks_h: int = 1, n_max_xticks: int = 15,
                          xtick_date_fmt: str = None, add_day_exp: bool = False) -> (List[int], List[str]):
    """
    Set xtick labels when x-axis is composed of dates
    :param x_dates: list of datetime of figure for which xticks must be set
    :param min_delta_xticks_h: min delta in hours between successive xtick labels
    :param n_max_xticks: max number of xtick labels
    :param xtick_date_fmt: month_in_letter -> Jan 1st; dow -> day of week
    :param add_day_exp: add day exponent (st for 1, nd for 2, etc.) if xtick_date_fmt is month_in_letter?
    """
    # TODO: to be set based on min delta xticks value (1h) and max nber of ticks
    idx_xticks = set_xtick_idx(min_date=x_dates[0], max_date=x_dates[-1], delta_date=x_dates[1] - x_dates[0],
                               min_delta_xticks_h=min_delta_xticks_h, n_max_xticks=n_max_xticks)
    if xtick_date_fmt is not None:
        if xtick_date_fmt == 'dow':
            xtick_labels = set_dow_xtick_labels(idx_xticks=idx_xticks, x_dates=x_dates)
        elif xtick_date_fmt == 'month_in_letter':
            xtick_labels = set_month_in_letter_xtick_labels(idx_xticks=idx_xticks, x_dates=x_dates,
                                                            add_day_exp=add_day_exp)
    return idx_xticks, xtick_labels


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

    # TODO: adapt
    # idx_xticks = np.arange(1, 1 + len(x_plot), fig_style.delta_xticks)
    # ax.set_xticks(idx_xticks)
    # if x_type == datetime:
    #     xtick_values = set_date_xtick_labels(curve_with_xdate=curves[0], idx_xticks=idx_xticks, fig_style=fig_style)
    # ax.set_xticklabels(xtick_values, rotation=fig_style.date_xtick_rotation, fontsize=fig_style.date_xtick_fontsize)

    plt.grid()
    if isinstance(y, dict) and with_curve_labels:
        plt.legend()
    plt.savefig(fig_file)
    plt.close()


if __name__ == '__main__':
    xticks = set_xtick_idx(min_date=datetime(2024, 1, 1), max_date=datetime(2024, 1, 8),
                           delta_date=timedelta(hours=2))
    bob = 1
