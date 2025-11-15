import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
import pandas as pd

from common.constants.data_analysis_types import COMMON_PLOT_YEAR
from common.plot_params import PlotParams
from utils.basic_utils import set_years_suffix, CLIM_YEARS_SUFFIX
from utils.dates import set_year_in_date, set_temporal_period_str
from utils.df_utils import set_key_columns
from utils.plot import simple_plot, set_temporal_period_title, FigureStyle, set_curve_style_attrs, CurveStyleAttrs

NAME_SEP = '_'
SUBNAME_SEP = '-'


def set_uc_ts_name(full_data_type: Tuple[str, List[str]], countries: List[str], years: List[int],
                   climatic_years: List[int], n_extra_params: int = None):
    """

    Args:
        full_data_type: (datatype, optionally list of subdt - only for RES CF currently)
        countries:
        years:
        climatic_years:
        n_extra_params:

    Returns:

    """
    # use only datatype for prefix
    full_dt_for_prefix = (full_data_type[0], )
    data_type_prefix = SUBNAME_SEP.join(list(full_dt_for_prefix))
    n_countries = len(countries)
    n_countries_max_in_suffix = 2
    n_countries_min_with_trigram = 2
    if n_countries_min_with_trigram <= n_countries <= n_countries_max_in_suffix:
        countries = [elt[:3] for elt in countries]
    countries_suffix = SUBNAME_SEP.join(countries) if n_countries <= n_countries_max_in_suffix \
        else SUBNAME_SEP.join([str(n_countries), 'countries'])
    years_suffix = set_years_suffix(years=years, sep=SUBNAME_SEP)
    clim_years_suffix = set_years_suffix(years=climatic_years, sep=SUBNAME_SEP, is_climatic_year=True)
    if CLIM_YEARS_SUFFIX not in clim_years_suffix:
        clim_years_suffix = f'cy{clim_years_suffix}'
    if n_extra_params is not None:
        extra_params_suffix = SUBNAME_SEP.join([str(n_extra_params), 'extraparams'])
    else:
        extra_params_suffix = ''
    # remove empty str
    suffix_lst = [data_type_prefix, countries_suffix, years_suffix, clim_years_suffix, extra_params_suffix]
    suffix_lst = [elt for elt in suffix_lst if len(elt) > 0]
    return NAME_SEP.join(suffix_lst)


def get_dims_from_uc_ts_name(name: str) -> Optional[Tuple[str, str, int, int]]:
    """
    Currently for case of unique element per dim (year for ex.); otherwise None returned
    """
    name_split = name.split(NAME_SEP)
    dims = []
    for i in range(4):
        if SUBNAME_SEP in name_split[i]:
            return None
        current_dim = int(name_split[i]) if name_split[i].isdigit() else name_split[i]
        dims.append(current_dim)
    return tuple(dims)


def set_curve_label(attrs_in_legend: List[str], country: str = None, year: int = None,
                    climatic_year: int = None, extra_args_label: str = None) -> str:
    sep = ', '
    label = ''
    if 'country' in attrs_in_legend and country is not None:
        label += country[:3]
    yr_labels = {'year': ('TY', year), 'climatic_year': ('CY', climatic_year)}
    for key, val in yr_labels.items():
        label_name = val[0]
        label_val = val[1]
        if key in attrs_in_legend and label_val is not None:
            if len(label) > 0:
                label += sep
            label += f'{label_name}={label_val}'
    if 'extra_args' in attrs_in_legend:
        label += extra_args_label if extra_args_label is not None else 'no extra-args'
    return label


def set_date_col(first_date: Union[int, datetime]) -> str:
    return 'time_slot' if isinstance(first_date, int) else 'date'


def set_y_with_label_as_key(y: Dict[tuple, Union[np.ndarray, list]], extra_params_labels: Dict[int, str],
                            attrs_in_legend: List[str]) -> Dict[tuple, Union[np.ndarray, list]]:
    """
    Set y-values from dict {case as tuple: values} to {label: values}
    """
    y_with_label = {}
    for key, vals in y.items():
        current_key = list(key)
        # replace last element of (tuple) key - the extra arg idx - by its label
        if current_key[-1] is not None:
            current_key[-1] = extra_params_labels[current_key[-1]]
        y_with_label[set_curve_label(attrs_in_legend, *current_key)] = vals
    return y_with_label


@dataclass
class UCTimeseries:
    name: str = None
    data_type: Tuple[str, List[str]] = None  # (datatype, list of sub-dts)
    # can be a dict. {(country, year, clim year): vector of values}, in case multiple
    # (country, year, climatic year) be considered
    values: Union[np.ndarray, Dict[Tuple[str, int, int], np.ndarray]] = None
    unit: str = None
    # can be a dict. {(country, year, clim year): dates}, in case multiple
    # (country, year, climatic year) be considered
    dates: Union[List[datetime], Dict[Tuple[str, int, int], List[datetime]]] = None

    def from_df_col(self, df: pd.DataFrame, col_name: str, unit: str = None):
        self.name = col_name
        self.values = np.array(df[col_name])
        if unit is not None:
            self.unit = unit

    def set_output_dates(self, is_plot: bool) -> Union[List[int], List[datetime]]:
        # per (country, year, clim year) values
        if isinstance(self.values, dict):
            first_key = list(self.values)[0]
            # repeat these ts index when saving a csv file, not when plotting (common x-axis)
            n_tile = len(self.values) if not is_plot else 1
        else:
            first_key = None
            n_tile = 1
        # dates as time-slots index
        if self.dates is None:
            if first_key is not None:
                vals_for_dates = self.values[first_key]
            else:
                vals_for_dates = self.values
            output_dates = np.tile(np.arange(len(vals_for_dates)) + 1, n_tile)
        # ... or dates (if provided)
        else:
            if first_key is not None:
                # saving to csv file -> concatenate the dates of all (country, year, clim year) cases
                if not is_plot:
                    output_dates = []
                    for key, dates_val in self.dates.items():
                        output_dates.extend(dates_val)
                else:
                    output_dates = self.dates[first_key]
                    # reset year to common values
                    output_dates = [set_year_in_date(my_date=elt, new_year=COMMON_PLOT_YEAR) for elt in output_dates]
            else:
                output_dates = self.dates
        return output_dates

    def set_output_values(self, is_plot: bool) -> Union[list, dict]:
        # per (country, year, clim year) values
        if isinstance(self.values, dict):
            # saving to csv file -> concatenate the values of all (country, year, clim year) cases
            if not is_plot:
                output_vals = []
                for key, vals in self.values.items():
                    output_vals.extend(vals)
            # plot -> dict.
            else:
                output_vals = self.values
        else:
            output_vals = self.values
        return output_vals

    def to_csv(self, output_dir: str, complem_columns: Dict[str, Union[list, np.ndarray, float]] = None,
               extra_params_labels: Dict[int, str] = None):
        """
        :param output_dir: in which csv must be saved
        :param complem_columns: to be added to saved csv
        :param extra_params_labels: {idx: label} corresp. for extra-parameters (no corresp. for None extra-params)
        """
        with_temp_period_suffix = True
        output_dates = self.set_output_dates(is_plot=False)
        date_col = set_date_col(first_date=output_dates[0])
        extra_params_col = 'extra_params'
        output_vals = self.set_output_values(is_plot=False)
        values_dict = {date_col: output_dates, 'value': output_vals}
        if complem_columns is not None:
            for col_name, col_vals in complem_columns.items():
                values_dict[col_name] = col_vals
        df_to_csv = pd.DataFrame(values_dict)
        # add "key" columns corresp. to the (country, ty, cy, extra-params) tuples
        if isinstance(self.dates, dict):
            all_keys = []
            # tuple keys of dates dict, replacing last component by label, if not None
            for elt_tuple in self.dates:
                if elt_tuple[-1] is None:
                    all_keys.append(elt_tuple)
                else:
                    all_keys.append(elt_tuple[:-1] + (extra_params_labels[elt_tuple[-1]],))
            n_dates = len(self.dates[all_keys[0]])
            df_keys = set_key_columns(col_names=['country', 'year', 'climatic_year', extra_params_col],
                                      tuple_values=all_keys, n_repeat=n_dates)
            df_to_csv = pd.concat([df_keys, df_to_csv], axis=1)
        if with_temp_period_suffix:
            min_date = min(output_dates)
            max_date = max(output_dates)
            temp_period_str = set_temporal_period_str(min_date=min_date, max_date=max_date,
                                                      print_year=False, date_sep='-')
            temp_period_suffix = f'_{temp_period_str}'
        else:
            temp_period_suffix = ''
        output_file = os.path.join(output_dir, f'{self.name.lower()}{temp_period_suffix}.csv')
        # remove extra-params column if unique value is None (i.e., no extra-params applied)
        if df_to_csv[extra_params_col].isna().all():
            del df_to_csv[extra_params_col]
        df_to_csv.to_csv(output_file, index=None)

    def set_plot_ylabel(self) -> str:
        ylabel = self.data_type[0].capitalize()
        if self.unit is not None:
            ylabel += f' ({self.unit.upper()})'
        return ylabel

    def set_plot_title(self, dt_suffix: str = None) -> str:
        plot_title = '-'.join(list(self.data_type)).capitalize()
        if dt_suffix is not None:
            plot_title += f' {dt_suffix}'
        # add suffix to indicate temporal period
        if isinstance(self.dates, list):
            dates_for_title = self.dates
        else:
            first_key = list(self.dates)[0]
            dates_for_title = self.dates[first_key]

        min_date = min(dates_for_title)
        max_date = max(dates_for_title)
        plot_title += f', period {set_temporal_period_title(min_date=min_date, max_date=max_date)}'
        return plot_title

    def set_attrs_in_plot_legend(self) -> List[str]:
        """
        Set attributes to be part of plot legend, the ones for which at least two different values have been used in
        data selection
        """
        if not isinstance(self.values, dict):
            return []
        all_tuples_in_vals = list(self.values)
        pot_attrs_for_plot_legend = {'country': 0, 'year': 1, 'climatic_year': 2, 'extra_args': 3}
        attrs_in_plot_legend = []
        for attr_name, attr_idx in pot_attrs_for_plot_legend.items():
            all_vals = set([elt[attr_idx] for elt in all_tuples_in_vals])
            if len(all_vals) > 1:
                attrs_in_plot_legend.append(attr_name)
        return attrs_in_plot_legend

    def set_curve_style_attrs(self, fig_style: FigureStyle = None, per_dim_plot_params: Dict[str, PlotParams] = None,
                              curve_labels: List[str] = None) \
            -> Optional[Union[CurveStyleAttrs, Dict[str, CurveStyleAttrs]]]:
        """
        Set curve styles attributes (color, linestyle, marker) for a plot
        :param fig_style: FigureStyle object to define some style attrs for plot
        :param per_dim_plot_params: per plot dimension (zone, year, climatic year) plot parameter values (the ones
        defined in plot_params.json file)
        :param curve_labels: list of curve labels, or None if unique curve
        :returns either an object CurveStyleAttrs (if unique curve on plot) or dict {curve label: CurveStyleAttrs}
        (if multiple ones)
        """
        if fig_style is not None and per_dim_plot_params is not None:
            if isinstance(self.values, dict):
                plot_dims_tuples = list(self.values)
            else:  # unique curve plotted -> get plot dimensions from UC ts name
                plot_dims_tuples = get_dims_from_uc_ts_name(name=self.name)[1:]
            curve_style_attrs = (
                set_curve_style_attrs(plot_dims_tuples=plot_dims_tuples,
                                      plot_dims_order=fig_style.plot_dims_order,
                                      per_dim_plot_params=per_dim_plot_params,
                                      curve_style=fig_style.curve_style)
            )
            # set same keys as the ones of y, i.e. labels
            if curve_labels is not None:
                curve_style_attrs_vals = list(curve_style_attrs.values())
                curve_style_attrs = {label_key: curve_style_attrs_vals[i] for i, label_key in enumerate(curve_labels)}
            else:  # y is a list; i.e. unique curve -> unique style attr. object
                curve_style_attrs = list(curve_style_attrs.values())[0]
        else:
            curve_style_attrs = None
        return curve_style_attrs

    def plot(self, output_dir: str, fig_style: FigureStyle = None, per_dim_plot_params: Dict[str, PlotParams] = None,
             extra_params_labels: Dict[int, str] = None):
        """
        Plot (UC) timeseries
        :param output_dir: in which figure will be saved
        :param fig_style: FigureStyle object to define some style attrs for plot
        :param per_dim_plot_params: per plot dimension (zone, year, climatic year) plot parameter values (the ones
        defined in plot_params.json file)
        :param extra_params_labels: corresp. between extra. parameters index and labels
        """
        name_label = self.name.capitalize()
        fig_file = os.path.join(output_dir, f'{name_label.lower()}.png')
        x = self.set_output_dates(is_plot=True)
        y = self.set_output_values(is_plot=True)
        xlabel = set_date_col(first_date=x[0]).capitalize() + 's'
        # replace (country, year, clim year, possibly extra-args label) keys by labels to be used for plot
        if isinstance(y, dict):
            attrs_in_legend = self.set_attrs_in_plot_legend()
            y = set_y_with_label_as_key(y=y, extra_params_labels=extra_params_labels, attrs_in_legend=attrs_in_legend)
        # set curve styles (color, linestyle, marker)
        curve_labels = list(y) if isinstance(y, dict) else None
        curve_style_attrs = self.set_curve_style_attrs(fig_style=fig_style, per_dim_plot_params=per_dim_plot_params,
                                                       curve_labels=curve_labels)
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simple_plot(x=x, y=y, fig_file=fig_file, title=self.set_plot_title(), xlabel=xlabel,
                        ylabel=self.set_plot_ylabel(), fig_style=fig_style, curve_style_attrs=curve_style_attrs)

    def plot_duration_curve(self, output_dir: str, as_a_percentage: bool = False, fig_style: FigureStyle = None,
                            per_dim_plot_params: Dict[str, PlotParams] = None,
                            extra_params_labels: Dict[int, str] = None):
        """
        Plot (UC) timeseries duration curve(s)
        :param output_dir: in which figure will be saved
        :param as_a_percentage: plot percentage values? (or in absolute values)
        :param fig_style: FigureStyle object to define some style attrs for plot
        :param per_dim_plot_params: per plot dimension (zone, year, climatic year) plot parameter values (the ones
        defined in plot_params.json file)
        :param extra_params_labels: corresp. between extra. parameters index and labels
        """
        y = self.set_output_values(is_plot=True)
        # sort values in descending order
        # per (country, year, climatic year) values
        if isinstance(y, dict):
            y_desc_order = {key: np.sort(vals)[::-1] for key, vals in y.items()}
            first_key = list(y)[0]
            n_vals = len(y_desc_order[first_key])
            attrs_in_legend = self.set_attrs_in_plot_legend()
            y_desc_order = set_y_with_label_as_key(y=y_desc_order, extra_params_labels=extra_params_labels,
                                                   attrs_in_legend=attrs_in_legend)
        else:
            y_desc_order = np.sort(y)[::-1]
            n_vals = len(y_desc_order)
        # this calculation is done assuming uniform time-slot duration
        duration_curve = np.arange(1, n_vals + 1)
        if as_a_percentage:
            duration_curve = np.cumsum(duration_curve) / len(duration_curve)
            xlabel = 'Duration (%)'
        else:
            xlabel = 'Duration (nber of time-slots - hours)'
        fig_file = os.path.join(output_dir, f'{self.name.lower()}_duration_curve.png')
        if fig_style is None:
            fig_style = FigureStyle()
            fig_style.process()
            print_legend = isinstance(y, dict) and len(y) > 1
            fig_style.set_print_legend(value=print_legend)
        # set curve styles (color, linestyle, marker)
        curve_labels = list(y_desc_order) if isinstance(y_desc_order, dict) else None
        curve_style_attrs = self.set_curve_style_attrs(fig_style=fig_style, per_dim_plot_params=per_dim_plot_params,
                                                       curve_labels=curve_labels)
        # catch DeprecationWarnings TODO: fix/more robust way to catch them?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simple_plot(x=duration_curve, y=y_desc_order, fig_file=fig_file,
                        title=f'{self.set_plot_title(dt_suffix="duration curve")}', xlabel=xlabel,
                        ylabel=self.set_plot_ylabel(), fig_style=fig_style, curve_style_attrs=curve_style_attrs)

    def plot_rolling_horizon_avg(self):
        bob = 1


def list_of_uc_timeseries_to_df(uc_timeseries: List[UCTimeseries]) -> pd.DataFrame:
    uc_ts_dict = {uc_ts.name: uc_ts.values for uc_ts in uc_timeseries}
    # add dates, if available
    if uc_timeseries[0].dates is not None:
        uc_ts_dict['date'] = uc_timeseries[0].dates
    return pd.DataFrame(uc_ts_dict)


# TODO: usage of this function?
def list_of_uc_ts_to_csv(list_of_uc_ts: List[UCTimeseries], output_dir: str, to_matrix_format: bool = False):
    # 1 file per UC timeseries
    if not to_matrix_format:
        for uc_ts in list_of_uc_ts:
            dummy_file = os.path.join(output_dir, 'dummy.csv')
            uc_ts.to_csv(dummy_file)
