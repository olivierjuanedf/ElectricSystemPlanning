import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union, Tuple
import numpy as np
import pandas as pd

from common.constants.data_analysis_types import ANALYSIS_TYPES_PLOT, COMMON_PLOT_YEAR
from utils.basic_utils import set_years_suffix
from utils.dates import set_year_in_date
from utils.plot import simple_plot


def set_uc_ts_name(full_data_type: tuple, countries: str, years: int, climatic_years: int):
    data_type_prefix = '-'.join(list(full_data_type))
    n_countries = len(countries)
    n_countries_max_in_suffix = 2
    n_countries_min_with_trigram = 2
    if n_countries_min_with_trigram <= n_countries <= n_countries_max_in_suffix:
        countries = [elt[:3] for elt in countries]
    countries_suffix = '-'.join(countries) if n_countries <= n_countries_max_in_suffix else f'{n_countries}-countries'
    years_suffix = set_years_suffix(years=years)
    clim_years_suffix = set_years_suffix(years=climatic_years)
    return f'{data_type_prefix}_{countries_suffix}_{years_suffix}_cy{clim_years_suffix}'


def set_curve_label(attrs_in_legend: List[str], country: str = None, year: int = None,
                    climatic_year: int = None) -> str:
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
    return label


def set_date_col(first_date: Union[int, datetime]) -> str:
    return 'time_slot' if isinstance(first_date, int) else 'date'


@dataclass
class UCTimeseries:
    name: str = None
    data_type: tuple = None
    # can be a dict. {(year, clim year): vector of values}, in case multiple (year, climatic year) be considered
    values: Union[np.ndarray, Dict[Tuple[int, int], np.ndarray]] = None
    unit: str = None
    # can be a dict. {(year, clim year): dates}, in case multiple (year, climatic year) be considered
    dates: Union[List[datetime], Dict[Tuple[int, int], List[datetime]]] = None

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

    def to_csv(self, output_dir: str, complem_columns: Dict[str, Union[list, np.ndarray, float]] = None):
        output_dates = self.set_output_dates(is_plot=False)
        date_col = set_date_col(first_date=output_dates[0])
        output_vals = self.set_output_values(is_plot=False)
        values_dict = {date_col: output_dates, 'value': output_vals}
        # TODO: add columns corresp. to the (country, ty, cy)
        if complem_columns is not None:
            for col_name, col_vals in complem_columns.items():
                values_dict[col_name] = col_vals
        df_to_csv = pd.DataFrame(values_dict)
        output_file = os.path.join(output_dir, f'{self.name.lower()}_uc-timeseries.csv')
        df_to_csv.to_csv(output_file)

    def set_plot_ylabel(self) -> str:
        ylabel = self.data_type[0].capitalize()
        if self.unit is not None:
            ylabel += f' ({self.unit.upper()})'
        return ylabel
    
    def set_plot_title(self) -> str:
        return '-'.join(list(self.data_type)).capitalize()

    def set_attrs_in_plot_legend(self) -> List[str]:
        if not isinstance(self.values, dict):
            return []
        all_tuples_in_vals = list(self.values)
        all_countries = set([elt[0] for elt in all_tuples_in_vals])
        all_years = set([elt[1] for elt in all_tuples_in_vals])
        all_clim_years = set([elt[2] for elt in all_tuples_in_vals])
        attrs_in_plot_legend = []
        if len(all_countries) > 1:
            attrs_in_plot_legend.append('country')
        if len(all_years) > 1:
            attrs_in_plot_legend.append('year')
        if len(all_clim_years) > 1:
            attrs_in_plot_legend.append('climatic_year')
        return attrs_in_plot_legend

    def plot(self, output_dir: str):
        name_label = self.name.capitalize()
        fig_file = os.path.join(output_dir, f'{name_label.lower()}.png')
        x = self.set_output_dates(is_plot=True)
        y = self.set_output_values(is_plot=True)
        xlabel = set_date_col(first_date=x[0]).capitalize() + 's'
        # replace (country, year, clim year) keys by labels to be used for plot
        if isinstance(y, dict):
            attrs_in_legend = self.set_attrs_in_plot_legend()
            y = {set_curve_label(attrs_in_legend, *key): vals for key, vals in y.items()}
        with_curve_labels = isinstance(y, dict) and len(y) > 1
        simple_plot(x=x, y=y, fig_file=fig_file, title=self.set_plot_title(), xlabel=xlabel,
                    ylabel=self.set_plot_ylabel(), with_curve_labels=with_curve_labels)

    def plot_duration_curve(self, output_dir: str, as_a_percentage: bool = False) -> np.ndarray:
        y = self.set_output_values(is_plot=True)
        # sort values in descending order
        # per (country, year, climatic year) values
        if isinstance(y, dict):
            vals_desc_order = {key: np.sort(vals)[::-1] for key, vals in y.items()}
            first_key = list(y)[0]
            n_vals = len(vals_desc_order[first_key])
            attrs_in_legend = self.set_attrs_in_plot_legend()
            vals_desc_order = {set_curve_label(attrs_in_legend, *key): vals
                               for key, vals in vals_desc_order.items()}
        else:
            vals_desc_order = np.sort(y)[::-1]
            n_vals = len(vals_desc_order)
        # this calculation is done assuming uniform time-slot duration
        duration_curve = np.arange(1, n_vals + 1)
        if as_a_percentage:
            duration_curve = np.cumsum(duration_curve) / len(duration_curve)
            xlabel = 'Duration (%)'
        else:
            xlabel = 'Duration (nber of time-slots - hours)'
        fig_file = os.path.join(output_dir, f'{self.name.lower()}_duration_curve.png')
        with_curve_labels = isinstance(y, dict) and len(y) > 1
        simple_plot(x=duration_curve, y=vals_desc_order, fig_file=fig_file,
                    title=f'{self.set_plot_title()} duration curve', xlabel=xlabel, 
                    ylabel=self.set_plot_ylabel(), with_curve_labels=with_curve_labels)
    
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

