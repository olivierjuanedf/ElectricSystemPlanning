import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union
import numpy as np
import pandas as pd

from long_term_uc.utils.plot import simple_plot


def set_uc_ts_name(full_data_type: tuple, country: str, year: int, climatic_year: int):
    data_type_prefix = '-'.join(list(full_data_type))
    return f'{data_type_prefix}_{country}_{year}_cy{climatic_year}'

    
@dataclass
class UCTimeseries:
    name: str = None
    data_type: tuple = None
    values: np.ndarray = None
    unit: str = None
    dates: List[datetime] = None

    def from_df_col(self, df: pd.DataFrame, col_name: str, unit: str = None):
        self.name = col_name
        self.values = np.array(df[col_name])
        if unit is not None:
            self.unit = unit

    def to_csv(self, output_dir: str, complem_columns: Dict[str, Union[list, np.ndarray]]):
        if self.dates is None:
            idx_col = 'time_slot'
            idx_range = np.arange(len(self.values)) + 1
        else:
            idx_col = 'date'
            idx_range = self.dates
        values_dict = {idx_col: idx_range, 'value': self.values}
        for col_name, col_vals in complem_columns.items():
            values_dict[col_name] = col_vals
        df_to_csv = pd.DataFrame(values_dict)
        

    def set_plot_ylabel(self) -> str:
        ylabel = self.data_type[0].capitalize()
        if self.unit is not None:
            ylabel += f' ({self.unit.upper()})'
        return ylabel
    
    def set_plot_title(self) -> str:
        return '-'.join(list(self.data_type)).capitalize()

    def plot(self, output_dir: str):
        name_label = self.name.capitalize()
        fig_file = os.path.join(output_dir, f'{name_label.lower()}.png')
        if self.dates is not None:
            x = self.dates
        else:
            x = np.arange(len(self.values)) + 1
        simple_plot(x=x, y=self.values, fig_file=fig_file, title=self.set_plot_title(), 
                    xlabel='Time-slots', ylabel=self.set_plot_ylabel())

    def plot_duration_curve(self, output_dir: str, as_a_percentage: bool = False) -> np.ndarray:
        # sort values in descending order
        vals_desc_order = np.sort(self.values)[::-1]
        # this calculation is done assuming uniform time-slot duration
        duration_curve = np.arange(1, len(vals_desc_order) + 1)
        if as_a_percentage:
            duration_curve = np.cumsum(duration_curve) / len(duration_curve)
            xlabel = 'Duration (%)'
        else:
            xlabel = 'Duration (nber of time-slots - hours)'
        name_label = self.name.capitalize()
        fig_file = os.path.join(output_dir, f'{name_label.lower()}_duration_curve.png')
        simple_plot(x=duration_curve, y=vals_desc_order, fig_file=fig_file,
                    title=f'{self.set_plot_title()} duration curve', xlabel=xlabel, 
                    ylabel=self.set_plot_ylabel())
    
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

