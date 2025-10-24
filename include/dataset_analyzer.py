import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Union

from common.constants.datatypes import DatatypesNames
from common.constants.extract_eraa_data import ERAADatasetDescr
from common.constants.temporal import DATE_FORMAT_IN_JSON, MAX_DATE_IN_DATA, N_DAYS_DATA_ANALYSIS_DEFAULT
from common.error_msgs import uncoherent_param_stop
from utils.type_checker import CheckerNames, apply_params_type_check


@dataclass
class AnalysisTypes: 
    calc: str = 'calc'
    # only extract data from data folder and put it in output folder
    extract: str = 'extract'
    # idem, put putting it on 'matricial format', with different climatic years in column
    extract_to_mat: str = 'extract_to_mat'
    plot: str = 'plot'  # simple plot
    plot_duration_curve: str = 'plot_duration_curve'  # duration curve plot
    plot_rolling_horizon_avg: str = 'plot_rolling_horizon_avg'  # rolling horizon avg plot


ANALYSIS_TYPES = AnalysisTypes()
AVAILABLE_ANALYSIS_TYPES = list(ANALYSIS_TYPES.__dict__.values())
AVAILABLE_DATA_TYPES = list(DatatypesNames.__annotations__.values())
DATA_SUBTYPE_KEY = 'data_subtype'  # TODO[Q2OJ]: cleaner way to set/get it?
RAW_TYPES_FOR_CHECK = {'analysis_type': CheckerNames.is_str, 'data_type': CheckerNames.is_str, 
                       'data_subtype': CheckerNames.is_str, 'country': CheckerNames.is_str_or_list_of_str,
                       'year': CheckerNames.is_int_or_list_of_int, 'climatic_year': CheckerNames.is_int_or_list_of_int}


@dataclass
class DataAnalysis:
    analysis_type: str
    data_type: str
    countries: Union[str, List[str]]  # in JSON can be str or list of str; when reading convert all to List[str]
    years: Union[int, List[int]]  # idem above with int or List[int]
    climatic_years: Union[int, List[int]]
    data_subtype: str = None
    period_start: datetime = None
    period_end: datetime = None

    def __repr__(self):
        repr_str = 'ERAA data analysis with params:'
        repr_str += f'\n- of type {self.analysis_type}'
        if self.data_subtype is not None:
            data_type_suffix = f', and sub-datatype {self.data_subtype}'
        else:
            data_type_suffix = ''
        repr_str += f'\n- for data type: {self.data_type}{data_type_suffix}'
        repr_str += f'\n- countries: {self.countries}'
        repr_str += f'\n- years: {self.years}'
        repr_str += f'\n- climatic years: {self.climatic_years}'
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
    
    def process(self):
        # country, year, climatic year attrs all to List[.]
        if isinstance(self.countries, str):
            self.countries = [self.countries]
        if isinstance(self.years, int):
            self.years = [self.years]
        if isinstance(self.climatic_years, int):
            self.climatic_years = [self.climatic_years]

        # default is full year
        if self.period_start is None:
            self.period_start = datetime(year=1900, month=1, day=1)
            self.period_end = datetime(year=1900, month=12, day=1)
            if self.period_end is not None:
                logging.warning(f'End of period {self.period_end} cannot be used as start not defined')
        else:
            self.period_start = datetime.strptime(self.period_start, DATE_FORMAT_IN_JSON) 
            if self.period_end is None:       
                self.period_end = (
                    min(MAX_DATE_IN_DATA, self.period_start + timedelta(days=N_DAYS_DATA_ANALYSIS_DEFAULT)))
            else:
                self.period_end = datetime.strptime(self.period_end, DATE_FORMAT_IN_JSON)

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

        # coherence of start and end period
        if self.period_end <= self.period_start:
            errors_list.append(f'{self.period_end.strftime(DATE_FORMAT_IN_JSON)} '
                               f'before {self.period_start.strftime(DATE_FORMAT_IN_JSON)}')

        # stop if any error
        if len(errors_list) > 0:
            uncoherent_param_stop(param_errors=errors_list)
        else:
            logging.info('Input data analysis PARAMETERS ARE COHERENT!')
            logging.info(f'ANALYSIS CAN START with parameters: {str(self)}')

    def get_full_datatype(self) -> tuple:
        if self.data_subtype is None:
            return (self.data_type,)
        else:
            return self.data_type, self.data_subtype
