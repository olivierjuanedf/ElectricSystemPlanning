from itertools import product

import numpy as np
import logging

from common.constants.datatypes import DATATYPE_NAMES, UNITS_PER_DT
from common.logger import init_logger, stop_logger
from common.long_term_uc_io import OUTPUT_DATA_ANALYSIS_FOLDER, OUTPUT_FOLDER_LT
from utils.basic_utils import get_period_str
from include.dataset import Dataset
from include.dataset_analyzer import ANALYSIS_TYPES
from include.uc_timeseries import UCTimeseries, set_uc_ts_name
from utils.read import read_and_check_data_analysis_params, read_and_check_uc_run_params

usage_params, eraa_data_descr, uc_run_params = read_and_check_uc_run_params()
data_analyses = read_and_check_data_analysis_params(eraa_data_descr=eraa_data_descr)

logger = init_logger(logger_dir=OUTPUT_FOLDER_LT, logger_name='eraa_input_data_analysis',
                     log_level=usage_params.log_level)
logging.info('START ERAA (input) data analysis')

uc_period_msg = get_period_str(period_start=uc_run_params.uc_period_start,
                               period_end=uc_run_params.uc_period_end)

# loop over the different cases to be analysed
for elt_analysis in data_analyses:
    logging.info(elt_analysis)
    # set UC run params to the ones corresponding to this analysis
    current_countries = elt_analysis.countries
    uc_run_params.set_countries(countries=current_countries)
    # currently loop over year, climatic_year; given that UC run params made for a unique (year, climatic year) couple
    # init. dict. to save data for each (country, year, clim_year) tuple
    current_df = {}
    for year, clim_year in product(elt_analysis.years, elt_analysis.climatic_years):
        uc_run_params.set_target_year(year=year)
        uc_run_params.set_climatic_year(climatic_year=clim_year)
        # Attention check at each time if stress test based on the set year
        uc_run_params.set_is_stress_test(avail_cy_stress_test=eraa_data_descr.available_climatic_years_stress_test)
        # And if coherent climatic year, i.e. in list of available data
        uc_run_params.coherence_check_ty_and_cy(eraa_data_descr=eraa_data_descr, stop_if_error=True)

        logging.info(f'Read needed ERAA ({eraa_data_descr.eraa_edition}) data for period {uc_period_msg}')
        # initialize dataset object
        eraa_dataset = Dataset(source=f'eraa_{eraa_data_descr.eraa_edition}',
                               agg_prod_types_with_cf_data=eraa_data_descr.agg_prod_types_with_cf_data,
                               is_stress_test=uc_run_params.is_stress_test)

        if elt_analysis.data_subtype is not None:
            subdt_selec = [elt_analysis.data_subtype]
        else:
            subdt_selec = None
        eraa_dataset.get_countries_data(uc_run_params=uc_run_params,
                                        aggreg_prod_types_def=eraa_data_descr.aggreg_prod_types_def,
                                        datatypes_selec=[elt_analysis.data_type], subdt_selec=subdt_selec)
        # create Unit Commitment Timeseries object from data read
        if elt_analysis.data_type == DATATYPE_NAMES.demand:
            # loop over country to extract per-country data from dataset.
            # N.B. year and climatic_year have been uniquely set up when init. the Dataset object
            for country in current_countries:
                current_df[(country, year, clim_year)] = eraa_dataset.demand[country]
        elif elt_analysis.data_type == DATATYPE_NAMES.capa_factor:
            # idem
            for country in current_countries:
                current_df[(country, year, clim_year)] = eraa_dataset.agg_cf_data[country]
        elif elt_analysis.data_type == DATATYPE_NAMES.net_demand:
            # idem
            for country in current_countries:
                current_df[(country, year, clim_year)] = eraa_dataset.net_demand[country]
        else:
            for country in current_countries:
                current_df[(country, year, clim_year)] = None
    elt_analysis.apply_analysis(per_case_data=current_df)

logging.info('THE END of ERAA (input) data analysis!')
stop_logger()
