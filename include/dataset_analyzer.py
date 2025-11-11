import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import List, Union, Dict, Tuple

import numpy as np
import pandas as pd

from common.constants.data_analysis_types import ANALYSIS_TYPES, ANALYSIS_TYPES_PLOT, AVAILABLE_ANALYSIS_TYPES
from common.constants.datatypes import DatatypesNames, UNITS_PER_DT
from common.constants.extract_eraa_data import ERAADatasetDescr, FICTIVE_CALENDAR_YEAR
from common.constants.temporal import DATE_FORMAT_IN_JSON, MAX_DATE_IN_DATA, N_DAYS_DATA_ANALYSIS_DEFAULT
from common.error_msgs import uncoherent_param_stop
from common.long_term_uc_io import OUTPUT_DATA_ANALYSIS_FOLDER
from common.plot_params import PlotParams
from include.uc_timeseries import set_uc_ts_name, UCTimeseries
from utils.basic_utils import random_draw_in_list
from utils.dates import robust_date_parser, set_year_in_date, set_temporal_period_str
from utils.plot import FigureStyle
from utils.type_checker import CheckerNames, apply_params_type_check


AVAILABLE_DATA_TYPES = list(DatatypesNames.__annotations__.values())
DATA_SUBTYPE_KEY = 'data_subtype'  # TODO[Q2OJ]: cleaner way to set/get it?
RAW_TYPES_FOR_CHECK = {'analysis_type': CheckerNames.is_str, 'data_type': CheckerNames.is_str, 
                       'data_subtype': CheckerNames.is_str, 'country': CheckerNames.is_str_or_list_of_str,
                       'year': CheckerNames.is_int_or_list_of_int, 'climatic_year': CheckerNames.is_int_or_list_of_int}
N_CURVES_MAX = 6
DEFAULT_CY = 'first'


def set_period_for_analysis(period_start: str, period_end: str) -> (datetime, datetime):
    # first try to cast provided dates to datetime
    if period_start is not None:
        period_start = robust_date_parser(my_date=period_start, raise_warning=True)
    if period_end is not None:
        period_end = robust_date_parser(my_date=period_end, raise_warning=True)
    # then set default values
    if period_start is None:
        period_start = datetime(year=1900, month=1, day=1)
    if period_end is None:
        if period_start is not None:
            logging.info(f'Period end set from start {period_start: %Y/%m/%d} and '
                         f'default number of days in period {N_DAYS_DATA_ANALYSIS_DEFAULT}')
            period_end = (
                min(MAX_DATE_IN_DATA, period_start + timedelta(days=N_DAYS_DATA_ANALYSIS_DEFAULT)))
        else:
            period_end = datetime(year=1900, month=12, day=31)
    return period_start, period_end


def set_period_to_common_year(period_start: datetime, period_end: datetime) -> (datetime, datetime):
    period_end_year = period_end.year
    period_start_year = period_start.year
    if not period_end_year == period_start_year:
        period_end_bis = datetime(year=period_start_year, month=period_end.month, day=period_end.day)
        if period_end_bis < period_start:
            period_start = (
                datetime(year=period_end_year, month=period_start.month, day=period_start.day)
            )
            common_year = period_end_year
            changed_extr = 'start'
        else:
            period_end = period_end_bis
            common_year = period_start_year
            changed_extr = 'end'
        logging.warning(f'Start and end of period do not share same year ({period_start_year} '
                        f'and {period_end_year} resp.) -> common year {common_year} will be considered, '
                        f'by changing {changed_extr} of period date')
    return period_start, period_end


def set_period_to_fixed_year(period_start: datetime, period_end: datetime, year: int) -> (datetime, datetime):
    # TODO: more robust code for this function (if not same calendar in period start/end years and in year...) ->
    #  reset start/end in year so that period has same nber of days
    period_start_year = period_start.year
    period_end_year = period_end.year
    if period_start_year == period_end_year and period_start_year == year:
        return period_start, period_end

    logging.info(f'{period_start:%Y/%m/%d} and {period_end:%Y/%m/%d} not in fixed year {year} -> modified')
    period_start = set_year_in_date(my_date=period_start, new_year=year)
    period_end = set_year_in_date(my_date=period_end, new_year=year)
    return period_start, period_end


def get_default_climatic_year(available_climatic_years: List[int]) -> int:
    if DEFAULT_CY == 'first':
        default_cy = min(available_climatic_years)
    elif DEFAULT_CY == 'last':
        default_cy = max(available_climatic_years)
    else:  # random draw
        default_cy = random_draw_in_list(my_list=available_climatic_years)
    return default_cy


def stop_if_coherence_check_error(obj_checked, errors_list: List[str]):
    # stop if any error
    if len(errors_list) > 0:
        uncoherent_param_stop(param_errors=errors_list)
    else:
        logging.info('Input data analysis PARAMETERS ARE COHERENT!')
        logging.info(f'ANALYSIS CAN START with parameters: {str(obj_checked)}')


@dataclass
class ExtraParamNames:
    capas_aggreg_pt_with_cf: str = 'capas_aggreg_pt_with_cf'


@dataclass
class DataAnalExtraParams:
    values: dict
    index: int = 1
    label: str = None

    def __repr__(self) -> str:
        repr_str = f'{self.label}: ' if self.label is not None else ''
        repr_str += str(self.values)
        return repr_str

    def process(self):
        if self.label is None:
            self.label = f'case {self.index}'

    def coherence_check(self, eraa_data_descr: ERAADatasetDescr):
        errors_list = []
        # check that names of extra-params values are coherent
        all_extra_param_names = set(ExtraParamNames.capas_aggreg_pt_with_cf)
        current_param_names = set(self.values)
        unknown_param_names = list(current_param_names - all_extra_param_names)
        if len(unknown_param_names) > 0:
            errors_list.append(f'Unknown extra-param names (keys of value dict.): {unknown_param_names}')

        # check that aggreg. prod. types are in allowed list
        fixed_cf_capas_key = ExtraParamNames.capas_aggreg_pt_with_cf
        if fixed_cf_capas_key in self.values:
            cf_capas_keys = list(self.values[fixed_cf_capas_key])
            unknown_cf_capas = [elt for elt in cf_capas_keys if elt not in eraa_data_descr.available_aggreg_prod_types]
            if len(unknown_cf_capas) > 0:
                errors_list.append(f'Unknown aggreg. prod. type with capa. factor data: {unknown_cf_capas}')

        # stop if any error
        stop_if_coherence_check_error(obj_checked=self, errors_list=errors_list)


@dataclass
class DataAnalysis:
    analysis_type: str
    data_type: str
    countries: Union[str, List[str]]  # in JSON can be str or list of str; when reading convert all to List[str]
    years: Union[int, List[int]]  # idem above with int or List[int]
    climatic_years: Union[int, List[int]]
    data_subtype: str = None
    period_start: Union[str, datetime] = None  # in JSON str, then datetime after parsing data
    period_end: Union[str, datetime] = None  # idem
    # Extra parameters to be used for data analysis, e.g. RES installed capacities - that can be used for
    # net demand calculation
    # in JSON dict or list of dicts, then list objects after parsing data
    extra_params: Union[dict, List[dict], List[DataAnalExtraParams]] = None

    def __repr__(self) -> str:
        sep_in_str = '\n- '
        repr_str = 'ERAA data analysis with params:'
        repr_str += f'{sep_in_str}of type {self.analysis_type}'
        if self.data_subtype is not None:
            data_type_suffix = f', and sub-datatype {self.data_subtype}'
        else:
            data_type_suffix = ''
        repr_str += f'{sep_in_str}for data type: {self.data_type}{data_type_suffix}'
        repr_str += f'{sep_in_str}countries: {self.countries}'
        repr_str += f'{sep_in_str}years: {self.years}'
        repr_str += f'{sep_in_str}climatic years: {self.climatic_years}'
        if self.period_start is not None and self.period_end is not None:
            temp_period_str = (
                set_temporal_period_str(min_date=self.period_start, max_date=self.period_end,
                                        print_year=False, in_letter=True)
            )
            repr_str += f'{sep_in_str}period: {temp_period_str}'
        if self.extra_params is not None:
            repr_str += f'{sep_in_str}extra-params: {str(self.extra_params)}'

        return repr_str

    def check_types(self):
        """
        Check coherence of types
        """
        dict_for_check = self.__dict__
        if self.data_subtype is None:
            del dict_for_check[DATA_SUBTYPE_KEY]
        apply_params_type_check(dict_for_check, types_for_check=RAW_TYPES_FOR_CHECK, 
                                param_name='Data analysis params - to set the calc./plot to be done')
    
    def process(self, eraa_data_descr: ERAADatasetDescr):
        # country, year, climatic year attrs all to List[.]
        if isinstance(self.countries, str):
            self.countries = [self.countries]
        if isinstance(self.years, int):
            self.years = [self.years]
        # set default climatic year for data analysis
        if self.climatic_years is None:
            default_cy = get_default_climatic_year(available_climatic_years=eraa_data_descr.available_climatic_years)
            logging.info(f'Default climatic year {default_cy} used (as not defined in DataAnalysis JSON file)')
            self.climatic_years = [default_cy]
        elif isinstance(self.climatic_years, int):
            self.climatic_years = [self.climatic_years]

        self.period_start, self.period_end = (
            set_period_for_analysis(period_start=self.period_start, period_end=self.period_end)
        )

        if self.extra_params is not None:
            if isinstance(self.extra_params, dict):
                self.extra_params = [self.extra_params]
            extra_params_obj = []
            for i, elt in enumerate(self.extra_params):
                if elt is not None:
                    index = i + 1
                    elt['index'] = index
                    params = DataAnalExtraParams(**elt)
                    params.process()
                    extra_params_obj.append(params)
                else:
                    extra_params_obj.append(None)
            self.extra_params = extra_params_obj
        else:
            self.extra_params = [None]

    def coherence_check(self, eraa_data_descr: ERAADatasetDescr):
        errors_list = []
        # check that analysis type is in the list of allowed values
        if self.analysis_type not in AVAILABLE_ANALYSIS_TYPES:
            errors_list.append(f'Unknown data analysis type: {self.analysis_type}')
        # check country
        unknown_countries = [elt for elt in self.countries if elt not in eraa_data_descr.available_countries]
        if len(unknown_countries) > 0:
            errors_list.append(f'Unknown selected countries: {unknown_countries}')

        # check TY and CY
        unknown_years = [elt for elt in self.years if elt not in eraa_data_descr.available_target_years]
        if len(unknown_years) > 0:
            errors_list.append(f'Unknown target years: {unknown_years}')
        unknown_clim_years = \
            [elt for elt in self.climatic_years if elt not in eraa_data_descr.available_climatic_years
             and elt not in eraa_data_descr.available_climatic_years_stress_test]
        if len(unknown_clim_years) > 0:
            errors_list.append(f'Unknown climatic years: {unknown_clim_years}')

        # check maximal nber of curves
        if self.analysis_type in ANALYSIS_TYPES_PLOT:
            n_curves = len(self.countries) * len(self.years) * len(self.climatic_years)
            if n_curves > N_CURVES_MAX:
                errors_list.append(f'Too many curves for {self.analysis_type}: {n_curves} '
                                   f'(vs. max allowed {N_CURVES_MAX})')

        # coherence of start and end period
        if self.period_end <= self.period_start:
            errors_list.append(f'Period end {self.period_end.strftime(DATE_FORMAT_IN_JSON)} '
                               f'before start {self.period_start.strftime(DATE_FORMAT_IN_JSON)}')

        # restrict to same calendar year to simplify following treatments -> 1900 as the unique one in "modeled" data
        self.period_start, self.period_end = (
            set_period_to_common_year(period_start=self.period_start, period_end=self.period_end)
        )
        self.period_start, self.period_end = (
            set_period_to_fixed_year(period_start=self.period_start, period_end=self.period_end,
                                     year=FICTIVE_CALENDAR_YEAR)
        )

        # stop if any error
        stop_if_coherence_check_error(obj_checked=self, errors_list=errors_list)

    def get_full_datatype(self) -> tuple:
        if self.data_subtype is None:
            return (self.data_type,)
        else:
            return self.data_type, self.data_subtype

    def get_extra_args_idx_to_label_corresp(self) -> Dict[int, str]:
        return {elt.index: elt.label for elt in self.extra_params if elt is not None}

    def apply_analysis(self, per_case_data: Dict[Tuple[str, int, int], pd.DataFrame], fig_style: FigureStyle = None,
                       per_dim_plot_params: Dict[str, PlotParams] = None, extra_params_labels: Dict[int, str] = None):
        """
        Apply 'analysis', either saving data to csv, or plotting it
        :param per_case_data: per tuple (country, year, climatic year) data in a dict. {tuple: df},
        or unique dataframe if unique case considered
        :param fig_style: FigureStyle params, in case a plot be applied
        :param per_dim_plot_params: {plot dimension eg 'zone': parameters to be used for plot color/linestyle/marker}
        :param extra_params_labels: {idx: label} corresp. for extra-parameters (no corresp. for None extra-params)
        """
        current_full_dt = self.get_full_datatype()
        date_col = 'date'
        value_col = 'value'
        n_extra_params = len(self.extra_params) if self.extra_params is not None else None
        uc_ts_name = set_uc_ts_name(full_data_type=current_full_dt, countries=self.countries, years=self.years,
                                    climatic_years=self.climatic_years, n_extra_params=n_extra_params)
        # loop over (country, year, clim_year) of this analysis
        dates = {}
        values = {}
        for country, year, clim_year, current_extra_params in (
                product(self.countries, self.years, self.climatic_years, self.extra_params)):
            try:
                extra_params_idx = current_extra_params.index if current_extra_params is not None else None
                current_dates = list(per_case_data[(country, year, clim_year, extra_params_idx)][date_col])
            except:
                logging.error(f'No dates obtained from data {self.data_type}, for (country, year, clim. year, '
                              f'extra-params idx) = ({country}, {year}, {clim_year}, {extra_params_idx}) '
                              f'-> not integrated in this data analysis')
                continue
            # if data available continue analysis (and plot)
            dates[(country, year, clim_year, extra_params_idx)] = \
                [elt_date.replace(year=year) for elt_date in current_dates]
            values[(country, year, clim_year, extra_params_idx)] = (
                np.array(per_case_data[(country, year, clim_year, extra_params_idx)][value_col]))

        uc_timeseries = UCTimeseries(name=uc_ts_name, data_type=current_full_dt, dates=dates,
                                     values=values, unit=UNITS_PER_DT[self.data_type])
        # And apply calc./plot... and other operations
        if len(values) == 0:
            logging.warning(f'No data obtained for type {self.data_type} -> analysis (plot/save to .csv) not done')
        elif self.analysis_type == ANALYSIS_TYPES.plot:
            uc_timeseries.plot(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER, fig_style=fig_style,
                               per_dim_plot_params=per_dim_plot_params, extra_params_labels=extra_params_labels)
        elif self.analysis_type == ANALYSIS_TYPES.plot_duration_curve:
            uc_timeseries.plot_duration_curve(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER, fig_style=fig_style,
                                              per_dim_plot_params=per_dim_plot_params,
                                              extra_params_labels=extra_params_labels)
        elif self.analysis_type in [ANALYSIS_TYPES.extract, ANALYSIS_TYPES.extract_to_mat]:
            to_matrix = True if self.analysis_type == ANALYSIS_TYPES.extract_to_mat else False
            # TODO[debug]: to_matrix_format not an arg of this method..., complem_columns missing...
            uc_timeseries.to_csv(output_dir=OUTPUT_DATA_ANALYSIS_FOLDER)
