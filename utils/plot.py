import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Dict, List, Tuple

from common.constants.datadims import DataDimensions
from common.constants.temporal import DAY_OF_WEEK
from common.plot_params import PlotParams, PLOT_DIMS_ORDER
from utils.basic_utils import lowest_common_multiple, get_first_level_with_multiple_vals
from utils.dates import set_temporal_period_str, add_day_exponent, set_month_short_in_date, remove_useless_zero_in_date


@dataclass
class XtickDateFormat:
    dow: str = 'dow'  # day of week H:
    # Year month in letter day H:, wo repeating (year, month, day) if idem to previous xtick
    in_letter: str = 'in_letter'


@dataclass
class CurveStyles:
    absolute: str = 'absolute'
    relative: str = 'relative'


DEFAULT_DATE_XTICK_FMT = XtickDateFormat.in_letter


@dataclass
class FigureStyle:
    size: Tuple[int, int] = (10, 6)
    marker: str = None
    grid_on: bool = True
    # curve style def -> 'absolute' to set up (color, linestyle, marker) based on (zone, year, clim year) value
    # whatever content of figure (other curves plotted); 'relative' to define it relatively
    curve_style: str = CurveStyles.absolute
    # all legend parameters
    print_legend: bool = True
    legend_font_size: int = 15
    legend_loc: str = 'best'
    # xtick (labels)
    delta_xticks: int = None
    date_xtick_fmt: str = XtickDateFormat.in_letter
    add_day_exp_in_date_xtick: bool = False
    rm_useless_zeros_in_date_xtick: bool = True
    date_xtick_fontsize: int = 12
    date_xtick_rotation: int = 45

    def set_print_legend(self, value: bool):
        self.print_legend = value

    def set_add_day_exp(self, value: bool):
        self.add_day_exp_in_date_xtick = value


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
    idx_xticks = np.arange(0, n_dates, delta_xticks_h).astype(np.int64)
    return list(idx_xticks)


def set_date_xtick_labels(idx_xticks: List[int], x_dates: List[datetime], format: str, short_months: bool = True,
                          add_day_exp: bool = False, rm_useless_zeros: bool = True) -> List[str]:
    """
    Set date xtick labels
    :param idx_xticks: idx of xtick labels that will be used in plot
    :param x_dates: all dates associated to points plotted
    :param format: format to be used for xtick labels, cf. XtickDateFormat
    :param short_months: use short names for months (Jan. i.o. January, etc.), only for 'in_letter' format
    :param add_day_exp: add exponent on day nber, only for 'in_letter' format
    :param rm_useless_zeros: remove useless zeros in dates nbers?
    """
    with_year_in_xticks = x_dates[-1].year > x_dates[0].year  # only used for format 'in_letter'
    new_date = None
    i = 0
    n_xticks = len(idx_xticks)
    xtick_labels = []
    while i < n_xticks:
        current_date = x_dates[idx_xticks[i]]
        current_day_date = datetime(year=current_date.year, month=current_date.month, day=current_date.day)

        # add dow/(year, month, day) only for first tick of this dow/(year, month, day)
        if new_date is None or not current_day_date == new_date:
            if format == XtickDateFormat.dow:
                current_dow = DAY_OF_WEEK[current_day_date.isoweekday() - 1]
                current_label = f"{current_dow}\n{x_dates[idx_xticks[i]]:%H:}"
            elif format == XtickDateFormat.in_letter:
                # new year
                if with_year_in_xticks and (new_date is None or current_day_date.year > new_date.year):
                    current_date_fmt = '%Y %B %d'
                # new month -> month, d label
                elif new_date is None or not current_day_date.month == new_date.month:
                    current_date_fmt = '%B %d'
                # new day
                else:
                    current_date_fmt = '%d'
                date_str = current_date.strftime(current_date_fmt)
                if short_months and 'B' in current_date_fmt:
                    date_str = set_month_short_in_date(date=date_str)

                if rm_useless_zeros:
                    date_str = remove_useless_zero_in_date(date=date_str, date_sep=' ')

                # add day exponent?
                if add_day_exp and len(date_str) > 0:
                    date_str = add_day_exponent(date=date_str)
                # add hours
                # new line if year or month in str
                if len(date_str) >= 3:
                    date_sep = '\n'
                else:
                    date_sep = ' '
                date_str += f'{date_sep}{current_date.hour}:'
                current_label = date_str
                # set new date as current one
                new_date = current_day_date
            else:
                current_label = None

        # only hours for the other dates
        else:
            current_label = f"{current_date:%H:}"

        xtick_labels.append(current_label)
        # move on to next xtick label
        i += 1
    return xtick_labels


def set_date_xtick_idx_and_labels(x_dates: List[datetime], min_delta_xticks_h: int = 1, n_max_xticks: int = 15,
                                  xtick_date_fmt: str = None, add_day_exp: bool = False,
                                  rm_useless_zeros: bool = True) -> (List[int], List[str]):
    """
    Set xtick labels when x-axis is composed of dates
    :param x_dates: list of datetime of figure for which xticks must be set
    :param min_delta_xticks_h: min delta in hours between successive xtick labels
    :param n_max_xticks: max number of xtick labels
    :param xtick_date_fmt: in_letter -> Jan 1st; dow -> day of week
    :param add_day_exp: add day exponent (st for 1, nd for 2, etc.) if xtick_date_fmt is month_in_letter?
    :param rm_useless_zeros: remove useless zeros in dates nbers (Jan. 1 i.o. Jan. 01)?
    """
    # set idx of xticks based on min delta xticks value and max nber of ticks
    idx_xticks = set_xtick_idx(min_date=x_dates[0], max_date=x_dates[-1], delta_date=x_dates[1] - x_dates[0],
                               min_delta_xticks_h=min_delta_xticks_h, n_max_xticks=n_max_xticks)
    if xtick_date_fmt is None:
        xtick_date_fmt = DEFAULT_DATE_XTICK_FMT
    xtick_labels = set_date_xtick_labels(idx_xticks=idx_xticks, x_dates=x_dates, format=xtick_date_fmt,
                                         add_day_exp=add_day_exp, rm_useless_zeros=rm_useless_zeros)
    return idx_xticks, xtick_labels


@dataclass
class CurveStyleAttrs:
    color: str = 'blue'
    linestyle: str = '-'
    marker: str = None


def set_curve_style_attrs(plot_dims_tuples: List[Tuple[str, int, int]], per_dim_plot_params: Dict[str, PlotParams],
                          curve_style: str) -> Dict[Tuple[str, int, int], CurveStyleAttrs]:
    """
    returns {(zone, year, clim. year): (color, linestyle, marker)}
    """
    all_curve_styles = [CurveStyles.absolute, CurveStyles.relative]
    if curve_style not in all_curve_styles:
        logging.warning(f'Unknown curve style {curve_style} -> curve style attributes cannot be set')
        return None
    # color from zone, linestyle from year, marker from climatic year - applying a "hierarchy"
    # TODO: merge cases
    linestyle_level = None
    marker_level = None
    if curve_style == CurveStyles.absolute:
        color_level = 0
        linestyle_level = 1
        marker_level = 2
    elif curve_style == CurveStyles.relative:
        # use only color to "distinguish" curves as unique one here
        if len(plot_dims_tuples) == 1:
            color_level = 0
        else:  # more than one curve
            # get level (in tuple) which will define the color
            color_level = get_first_level_with_multiple_vals(tuple_list=plot_dims_tuples)
            if color_level < 2:
                linestyle_level = get_first_level_with_multiple_vals(tuple_list=plot_dims_tuples,
                                                                     init_level=color_level,
                                                                     return_none_if_not_found=True)
            if linestyle_level is not None and linestyle_level < 2:
                marker_level = get_first_level_with_multiple_vals(tuple_list=plot_dims_tuples,
                                                                  init_level=linestyle_level,
                                                                  return_none_if_not_found=True)

    # get dicts {plot dim value: style attr value} to be used
    per_case_color = per_dim_plot_params[PLOT_DIMS_ORDER[color_level]].per_case_color
    if linestyle_level is not None:
        per_case_linestyle = per_dim_plot_params[PLOT_DIMS_ORDER[linestyle_level]].per_case_linestyle
    if marker_level is not None:
        per_case_marker = per_dim_plot_params[PLOT_DIMS_ORDER[marker_level]].per_case_marker
    per_case_curve_style_attrs = {}
    for case_tuple in plot_dims_tuples:
        style_attrs_dict = {'color': per_case_color[case_tuple[color_level]]}
        if linestyle_level is not None:
            style_attrs_dict['linestyle'] = per_case_linestyle[case_tuple[linestyle_level]]
        if marker_level is not None:
            style_attrs_dict['marker'] = per_case_marker[case_tuple[marker_level]]
        per_case_curve_style_attrs[case_tuple] = CurveStyleAttrs(**style_attrs_dict)
    return per_case_curve_style_attrs


def simple_plot(x: Union[np.ndarray, list], y: Union[np.ndarray, list, Dict[str, np.ndarray], Dict[str, list]],
                fig_file: str, title: str, xlabel: str, ylabel: str, fig_style: FigureStyle = None):
    if fig_style is None:
        fig_style = FigureStyle()

    plt.figure(figsize=fig_style.size)
    # TODO: merge all cases in a unique call de plt.plot
    if isinstance(y, dict):
        for key_label, values in y.items():
            current_label = key_label if fig_style.print_legend else None
            plt.plot(x, values, marker=fig_style.marker, label=current_label)
    else:
        plt.plot(x, y, marker=fig_style.marker)

    # title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # add xtick date labels
    first_x = x[0]
    if isinstance(first_x, datetime):
        x_dates = x if isinstance(x, list) else list(x)
        idx_xticks, xtick_values = (
            set_date_xtick_idx_and_labels(x_dates=x_dates, xtick_date_fmt=fig_style.date_xtick_fmt,
                                          add_day_exp=fig_style.add_day_exp_in_date_xtick,
                                          rm_useless_zeros=fig_style.rm_useless_zeros_in_date_xtick)
        )
        ticks = [x_dates[i] for i in idx_xticks]
        plt.xticks(ticks=ticks, labels=xtick_values, rotation=fig_style.date_xtick_rotation,
                   fontsize=fig_style.date_xtick_fontsize)

    # grid
    if fig_style.grid_on:
        plt.grid()
    # legend
    if isinstance(y, dict) and fig_style.print_legend:
        plt.legend(loc=fig_style.legend_loc, fontsize=fig_style.legend_font_size)

    # save and close figure
    plt.savefig(fig_file)
    plt.close()
